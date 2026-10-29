from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .environment import LinearBanditTrial
from .linear_algebra import FrozenRidgeState


@dataclass(frozen=True)
class AlgorithmTrace:


    algorithm: str
    checkpoints: np.ndarray
    group_pseudo_regret: np.ndarray
    forced_exploration_group_pseudo_regret: np.ndarray
    policy_group_pseudo_regret: np.ndarray
    per_client_pseudo_regret: np.ndarray
    per_client_forced_exploration_pseudo_regret: np.ndarray
    per_client_policy_pseudo_regret: np.ndarray
    communication_bits: np.ndarray
    pulls_per_client: int
    metadata: dict[str, object]

    @property
    def final_group_pseudo_regret(self) -> float:
        return float(self.group_pseudo_regret[-1])

    @property
    def final_forced_exploration_group_pseudo_regret(self) -> float:


        return float(self.forced_exploration_group_pseudo_regret[-1])

    @property
    def final_policy_group_pseudo_regret(self) -> float:


        return float(self.policy_group_pseudo_regret[-1])

    @property
    def final_max_client_pseudo_regret(self) -> float:
        return float(np.max(self.per_client_pseudo_regret[-1]))

    @property
    def final_communication_bits(self) -> int:
        return int(self.communication_bits[-1])

    def summary(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "final_group_pseudo_regret": self.final_group_pseudo_regret,
            "final_forced_exploration_group_pseudo_regret": self.final_forced_exploration_group_pseudo_regret,
            "final_policy_group_pseudo_regret": self.final_policy_group_pseudo_regret,
            "final_max_client_pseudo_regret": self.final_max_client_pseudo_regret,
            "final_communication_bits": self.final_communication_bits,
            "pulls_per_client": self.pulls_per_client,
            "metadata": self.metadata,
        }


@dataclass
class _Recorder:


    config: ExperimentConfig
    cumulative_client_regret: np.ndarray
    cumulative_forced_client_regret: np.ndarray
    cumulative_policy_client_regret: np.ndarray
    checkpoints: list[int]
    group_values: list[float]
    forced_group_values: list[float]
    policy_group_values: list[float]
    client_values: list[np.ndarray]
    forced_client_values: list[np.ndarray]
    policy_client_values: list[np.ndarray]
    communication_values: list[int]

    @classmethod
    def create(cls, config: ExperimentConfig) -> "_Recorder":
        return cls(
            config=config,
            cumulative_client_regret=np.zeros(config.num_clients),
            cumulative_forced_client_regret=np.zeros(config.num_clients),
            cumulative_policy_client_regret=np.zeros(config.num_clients),
            checkpoints=[],
            group_values=[],
            forced_group_values=[],
            policy_group_values=[],
            client_values=[],
            forced_client_values=[],
            policy_client_values=[],
            communication_values=[],
        )

    def add_regret(
        self, client: int, pseudo_regret: float, *, forced_exploration: bool
    ) -> None:
        self.cumulative_client_regret[client] += pseudo_regret
        if forced_exploration:
            self.cumulative_forced_client_regret[client] += pseudo_regret
        else:
            self.cumulative_policy_client_regret[client] += pseudo_regret

    def maybe_record(
        self,
        completed_pulls: int,
        *,
        communication_bits: int,
    ) -> None:
        is_checkpoint = completed_pulls % self.config.checkpoint_interval == 0
        is_final = completed_pulls == self.config.horizon
        if not (is_checkpoint or is_final):
            return
        self.checkpoints.append(completed_pulls)
        self.group_values.append(float(np.sum(self.cumulative_client_regret)))
        self.forced_group_values.append(
            float(np.sum(self.cumulative_forced_client_regret))
        )
        self.policy_group_values.append(
            float(np.sum(self.cumulative_policy_client_regret))
        )
        self.client_values.append(self.cumulative_client_regret.copy())
        self.forced_client_values.append(self.cumulative_forced_client_regret.copy())
        self.policy_client_values.append(self.cumulative_policy_client_regret.copy())
        self.communication_values.append(int(communication_bits))

    def finish(
        self,
        *,
        algorithm: str,
        communication_bits: int,
        metadata: dict[str, object],
    ) -> AlgorithmTrace:
        if not self.checkpoints or self.checkpoints[-1] != self.config.horizon:
            self.maybe_record(
                self.config.horizon,
                communication_bits=communication_bits,
            )


        if self.communication_values:
            self.communication_values[-1] = int(communication_bits)
        checkpoints = np.asarray(self.checkpoints, dtype=int)
        group_values = np.asarray(self.group_values, dtype=float)
        forced_group_values = np.asarray(self.forced_group_values, dtype=float)
        policy_group_values = np.asarray(self.policy_group_values, dtype=float)
        if not np.allclose(
            group_values, forced_group_values + policy_group_values, atol=1e-12
        ):
            raise RuntimeError("pseudo-regret decomposition does not sum to total regret")
        client_values = np.asarray(self.client_values, dtype=float)
        forced_client_values = np.asarray(self.forced_client_values, dtype=float)
        policy_client_values = np.asarray(self.policy_client_values, dtype=float)
        if not np.allclose(
            client_values, forced_client_values + policy_client_values, atol=1e-12
        ):
            raise RuntimeError("per-client pseudo-regret decomposition does not sum to total")
        communication_values = np.asarray(self.communication_values, dtype=np.int64)
        for array in (
            checkpoints,
            group_values,
            forced_group_values,
            policy_group_values,
            client_values,
            forced_client_values,
            policy_client_values,
            communication_values,
        ):
            array.setflags(write=False)
        return AlgorithmTrace(
            algorithm=algorithm,
            checkpoints=checkpoints,
            group_pseudo_regret=group_values,
            forced_exploration_group_pseudo_regret=forced_group_values,
            policy_group_pseudo_regret=policy_group_values,
            per_client_pseudo_regret=client_values,
            per_client_forced_exploration_pseudo_regret=forced_client_values,
            per_client_policy_pseudo_regret=policy_client_values,
            communication_bits=communication_values,
            pulls_per_client=self.config.horizon,
            metadata=metadata,
        )


def _choose_action(
    trial: LinearBanditTrial,
    config: ExperimentConfig,
    *,
    state: FrozenRidgeState,
    features: np.ndarray,
    alpha: float,
    client: int,
    time_index: int,
) -> tuple[int, bool]:
    if (
        trial.exploration_uniforms[client, time_index]
        < config.effective_forced_exploration(time_index)
    ):
        return int(trial.exploration_actions[client, time_index]), True
    return int(np.argmax(state.ucb_scores(features, alpha))), False


def _epoch_ended(completed_pulls: int, config: ExperimentConfig) -> bool:
    return (
        completed_pulls % config.epoch_length == 0
        or completed_pulls == config.horizon
    )


def _validate_trial(trial: LinearBanditTrial, config: ExperimentConfig) -> None:
    config.validate()
    expected_context_shape = (config.num_clients, config.horizon)
    expected_noise_shape = (config.num_clients, config.horizon, config.num_actions)
    if trial.context_ids.shape != expected_context_shape:
        raise ValueError("trial context tape does not match configuration")
    if trial.reward_noise.shape != expected_noise_shape:
        raise ValueError("trial reward tape does not match configuration")
    if trial.theta_global.shape != (config.global_dim,):
        raise ValueError("trial global parameter does not match configuration")
    if trial.delta_local.shape != (config.num_clients, config.local_dim):
        raise ValueError("trial local parameters do not match configuration")
