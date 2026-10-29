from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codec import ProgressiveResidualTransport
from .config import ExperimentConfig
from .environment import LinearBanditTrial
from .graph import StaticGraph, make_static_graph
from .linear_algebra import (
    FrozenRidgeState,
    add_local_moment,
    client_state_from_reduced_statistics,
    concatenate_features,
    denormalize_reduced_payload,
    initialise_local_moments,
    local_reduced_statistics,
    normalize_reduced_payload,
    pack_reduced_statistics,
    reduced_payload_dimension,
    unpack_reduced_statistics,
)
from .trace import AlgorithmTrace, _Recorder, _choose_action, _epoch_ended, _validate_trial


@dataclass(frozen=True)
class FedRCBTrace:
    base: AlgorithmTrace
    flag_bits: np.ndarray
    residual_code_bits: np.ndarray
    scheduled_scalar_slots: np.ndarray
    active_residual_events: np.ndarray
    range_violation_count: np.ndarray
    packet_digest: str
    final_theta_estimates: np.ndarray

    @property
    def checkpoints(self) -> np.ndarray:
        return self.base.checkpoints

    @property
    def group_pseudo_regret(self) -> np.ndarray:
        return self.base.group_pseudo_regret

    @property
    def forced_exploration_group_pseudo_regret(self) -> np.ndarray:
        return self.base.forced_exploration_group_pseudo_regret

    @property
    def policy_group_pseudo_regret(self) -> np.ndarray:
        return self.base.policy_group_pseudo_regret

    @property
    def per_client_pseudo_regret(self) -> np.ndarray:
        return self.base.per_client_pseudo_regret

    @property
    def per_client_forced_exploration_pseudo_regret(self) -> np.ndarray:
        return self.base.per_client_forced_exploration_pseudo_regret

    @property
    def per_client_policy_pseudo_regret(self) -> np.ndarray:
        return self.base.per_client_policy_pseudo_regret

    @property
    def communication_bits(self) -> np.ndarray:
        return self.base.communication_bits


def run_fedrcb(
    trial: LinearBanditTrial,
    config: ExperimentConfig,
    graph: StaticGraph | None = None,
) -> FedRCBTrace:
    _validate_trial(trial, config)
    graph = graph or make_static_graph(config.topology, config.num_clients)
    if graph.num_clients != config.num_clients:
        raise ValueError("graph client count does not match configuration")
    codec_seed = progressive_codec_seed(
        config.codec_seed,
        trial_seed=trial.seed,
        width=config.bit_width,
        threshold_factor=config.threshold_factor,
    )
    alpha = config.confidence_radius(0)
    global_design, cross_design, local_design, global_response, local_response = (
        initialise_local_moments(config.num_clients, config.global_dim, config.local_dim)
    )
    initial_payloads = np.zeros(
        (config.num_clients, reduced_payload_dimension(config.global_dim)), dtype=float
    )
    states = _states_from_aggregate_payloads(
        initial_payloads, cross_design, local_design, local_response, config
    )
    transport = ProgressiveResidualTransport(
        graph,
        rounds=config.gossip_rounds,
        bit_width=config.bit_width,
        codec_seed=codec_seed,
        threshold_factor=config.threshold_factor,
        tolerance=max(config.cache_tolerance, config.normalization_tolerance),
    )
    recorder = _Recorder.create(config)
    ledger_values = {
        name: []
        for name in (
            "flag_bits", "residual_code_bits", "scheduled_scalar_slots",
            "active_residual_events", "range_violation_count",
        )
    }

    for time_index in range(config.horizon):
        for client in range(config.num_clients):
            xg, xl = trial.slate(client, time_index)
            features = concatenate_features(xg, xl)
            action, forced = _choose_action(
                trial, config, state=states[client], features=features,
                alpha=alpha, client=client, time_index=time_index,
            )
            reward = float(np.clip(
                trial.reward(client, time_index, action),
                -config.reward_clip, config.reward_clip,
            ))
            add_local_moment(
                global_design, cross_design, local_design, global_response,
                local_response, client=client, x_global=xg[action],
                x_local=xl[action], reward=reward,
            )
            recorder.add_regret(
                client, trial.pseudo_regret(client, time_index, action),
                forced_exploration=forced,
            )

        completed = time_index + 1
        if _epoch_ended(completed, config):
            local_reduced_design, local_reduced_response = local_reduced_statistics(
                global_design, cross_design, local_design, global_response,
                local_response, config.regularization,
            )
            local_payloads = pack_reduced_statistics(
                local_reduced_design, local_reduced_response
            )
            normalized = np.stack([
                normalize_reduced_payload(
                    local_payloads[client], global_dim=config.global_dim,
                    completed_pulls=completed, reward_clip=config.reward_clip,
                )
                for client in range(config.num_clients)
            ])
            normalized_max = float(np.max(np.abs(normalized))) if normalized.size else 0.0
            if normalized_max > 1.0 + config.normalization_tolerance:
                raise ValueError(
                    f"normalized Schur payload exceeded its public cube: {normalized_max:g}"
                )
            epoch_index = (completed - 1) // config.epoch_length + 1
            output = transport.run_epoch(normalized, epoch=epoch_index)
            aggregate_payloads = np.stack([
                config.num_clients * denormalize_reduced_payload(
                    output[client], global_dim=config.global_dim,
                    completed_pulls=completed, reward_clip=config.reward_clip,
                )
                for client in range(config.num_clients)
            ])
            states = _states_from_aggregate_payloads(
                aggregate_payloads, cross_design, local_design, local_response, config
            )
            alpha = config.confidence_radius(completed)

        recorder.maybe_record(completed, communication_bits=transport.ledger.payload_bits)
        if completed % config.checkpoint_interval == 0 or completed == config.horizon:
            for name, values in ledger_values.items():
                values.append(int(getattr(transport.ledger, name)))

    metadata = {
        "decision_protocol": "frozen epoch UCB with decaying uniform exploration",
        "parameterization": "shared global ridge and private local ridge blocks",
        "network_communication": "persistent residual-coded peer-to-peer gossip",
        "topology": graph.topology,
        "undirected_edges": graph.undirected_edge_count,
        "directed_edges": graph.directed_edge_count,
        "consensus_slem": graph.consensus_slem,
        "gossip_rounds_per_epoch": config.gossip_rounds,
        "scalar_slots_per_message": reduced_payload_dimension(config.global_dim),
        "residual_code_bits": config.bit_width,
        "flag_bits_per_slot": 1,
        "threshold_factor": config.threshold_factor,
        "codec_seed": codec_seed,
        "confidence_multiplier_initial": config.confidence_radius(0),
        "confidence_multiplier_final": alpha,
        "reward_clip": config.reward_clip,
    }
    base_trace = recorder.finish(
        algorithm="FedRCB", communication_bits=transport.ledger.payload_bits,
        metadata=metadata,
    )
    frozen_ledger = {}
    for name, values in ledger_values.items():
        array = np.asarray(values, dtype=np.int64)
        array.setflags(write=False)
        frozen_ledger[name] = array
    estimates = np.stack([state.theta for state in states])
    estimates.setflags(write=False)
    return FedRCBTrace(
        base=base_trace, packet_digest=transport.ledger.digest,
        final_theta_estimates=estimates, **frozen_ledger,
    )


def _states_from_aggregate_payloads(
    payloads: np.ndarray,
    cross_design: np.ndarray,
    local_design: np.ndarray,
    local_response: np.ndarray,
    config: ExperimentConfig,
) -> list[FrozenRidgeState]:
    states = []
    for client in range(config.num_clients):
        design, response = unpack_reduced_statistics(payloads[client], config.global_dim)
        states.append(client_state_from_reduced_statistics(
            design, response, local_cross_design=cross_design[client],
            local_design=local_design[client], local_response=local_response[client],
            regularization=config.regularization,
        ))
    return states


def progressive_codec_seed(
    codec_seed: int, *, trial_seed: int, width: int, threshold_factor: float
) -> int:
    gamma_key = int(round(float(threshold_factor) * 1_000_000))
    sequence = np.random.SeedSequence([
        int(codec_seed), int(trial_seed), int(width), int(gamma_key)
    ])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])
