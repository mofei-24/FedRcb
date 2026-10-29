from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .config import ExperimentConfig
from .context import (
    ClientContextDiagnostics,
    PublicClientContextProfile,
    PublicContextLibrary,
    context_sampling_probabilities,
    context_diagnostics_by_client,
    generate_public_client_context_profile,
    generate_public_context_library,
)


@dataclass(frozen=True)
class LinearBanditTrial:


    library: PublicContextLibrary
    theta_global: np.ndarray
    delta_local: np.ndarray
    context_ids: np.ndarray
    reward_noise: np.ndarray
    exploration_uniforms: np.ndarray
    exploration_actions: np.ndarray
    seed: int
    heterogeneity: float
    digest: str
    client_context_profile: PublicClientContextProfile | None = None
    client_context_diagnostics: ClientContextDiagnostics | None = None
    client_context_uniforms: np.ndarray | None = None

    @property
    def num_clients(self) -> int:
        return int(self.context_ids.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.context_ids.shape[1])

    @property
    def context_digest(self) -> str:


        return self.library.digest

    @property
    def context_profile_digest(self) -> str | None:


        if (
            self.client_context_profile is None
            or not self.client_context_profile.is_shifted
        ):
            return None
        return self.client_context_profile.digest

    @property
    def context_diagnostics(self) -> dict[str, float]:


        return self.library.diagnostics.as_dict()

    @property
    def client_context_diagnostic_summary(self) -> dict[str, float] | None:


        if (
            self.client_context_profile is None
            or not self.client_context_profile.is_shifted
            or self.client_context_diagnostics is None
        ):
            return None
        return self.client_context_diagnostics.as_dict()

    @property
    def context_distribution(self) -> str:


        if (
            self.client_context_profile is not None
            and self.client_context_profile.is_shifted
        ):
            profile = self.client_context_profile
            if profile.kind == "dirichlet":
                return (
                    "public finite client-specific Dirichlet catalogue "
                    f"(concentration={profile.concentration:g}, "
                    f"profile_seed={profile.profile_seed})"
                )
            if profile.kind == "movielens":
                return (
                    "public finite MovieLens-1M catalogue "
                    "(client laws = real cluster rating masses over "
                    "genre-anchored slates; surrogate ground truth)"
                )
            return f"public finite {profile.kind} catalogue"
        if self.library.is_uniform:
            return "public finite uniform catalogue"
        return "public finite nonuniform catalogue"

    def slate(self, client: int, time_index: int) -> tuple[np.ndarray, np.ndarray]:


        context_id = int(self.context_ids[client, time_index])
        return self.library.slate(context_id)

    def expected_rewards(self, client: int, time_index: int) -> np.ndarray:


        x_global, x_local = self.slate(client, time_index)
        return x_global @ self.theta_global + x_local @ self.delta_local[client]

    def reward(self, client: int, time_index: int, action: int) -> float:
        return float(
            self.expected_rewards(client, time_index)[action]
            + self.reward_noise[client, time_index, action]
        )

    def pseudo_regret(self, client: int, time_index: int, action: int) -> float:
        means = self.expected_rewards(client, time_index)
        return float(np.max(means) - means[action])


def generate_trial(
    config: ExperimentConfig,
    *,
    heterogeneity: float,
    seed: int,
    catalogue_seed: int | None = None,
) -> LinearBanditTrial:


    config.validate()
    if heterogeneity < 0:
        raise ValueError("heterogeneity must be non-negative")
    public_catalogue_seed = catalogue_seed
    if public_catalogue_seed is not None:
        if isinstance(public_catalogue_seed, bool) or not isinstance(
            public_catalogue_seed, (int, np.integer)
        ):
            raise ValueError("catalogue_seed must be a non-negative integer or None")
        if int(public_catalogue_seed) < 0:
            raise ValueError("catalogue_seed must be a non-negative integer or None")
    seed_sequence = np.random.SeedSequence(seed)
    generated_catalogue_seed, parameter_seed, tape_seed = seed_sequence.spawn(3)
    catalogue_rng = np.random.default_rng(
        generated_catalogue_seed
        if public_catalogue_seed is None
        else int(public_catalogue_seed)
    )
    parameter_rng = np.random.default_rng(parameter_seed)
    tape_rng = np.random.default_rng(tape_seed)
    base_context_probabilities = context_sampling_probabilities(config.library_size)
    client_context_profile = generate_public_client_context_profile(
        num_clients=config.num_clients,
        library_size=config.library_size,
        profile_seed=config.client_context_profile_seed,
        base_probabilities=base_context_probabilities,
        concentration=config.client_context_concentration,
    )
    library = generate_public_context_library(
        library_size=config.library_size,
        num_actions=config.num_actions,
        global_dim=config.global_dim,
        local_dim=config.local_dim,
        context_norm=config.context_norm,
        min_eigenvalue=config.min_context_eigenvalue,
        rng=catalogue_rng,
        required_client_sampling_probabilities=(
            client_context_profile.sampling_probabilities
            if client_context_profile.is_shifted
            else None
        ),
        local_energy_fraction=config.local_energy_fraction,
    )
    client_context_diagnostics = context_diagnostics_by_client(
        library.x_global,
        library.x_local,
        sampling_probabilities=client_context_profile.sampling_probabilities,
        base_probabilities=library.sampling_probabilities,
    )
    theta_global = _normalised_vector(
        parameter_rng.normal(size=config.global_dim), config.global_theta_norm
    )
    delta_local = _centred_local_effects(
        parameter_rng.normal(size=(config.num_clients, config.local_dim)), heterogeneity
    )
    tape_rng.integers(
        0,
        config.library_size,
        size=(config.num_clients, config.horizon),
        endpoint=False,
    )
    client_context_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 908_117])
    )
    client_context_uniforms = client_context_rng.random(
        size=(config.num_clients, config.horizon)
    )
    context_ids = _sample_client_context_ids(
        client_context_profile.sampling_probabilities,
        client_context_uniforms,
    )
    reward_noise = tape_rng.normal(
        loc=0.0,
        scale=config.noise_std,
        size=(config.num_clients, config.horizon, config.num_actions),
    )
    exploration_uniforms = tape_rng.random(size=(config.num_clients, config.horizon))
    exploration_actions = tape_rng.integers(
        0,
        config.num_actions,
        size=(config.num_clients, config.horizon),
        endpoint=False,
    )
    for array in (
        theta_global,
        delta_local,
        context_ids,
        reward_noise,
        exploration_uniforms,
        exploration_actions,
        *(() if client_context_uniforms is None else (client_context_uniforms,)),
    ):
        array.setflags(write=False)
    digest = _trial_digest(
        library.digest,
        theta_global,
        delta_local,
        context_ids,
        reward_noise,
        exploration_uniforms,
        exploration_actions,
        context_profile_digest=(
            client_context_profile.digest if client_context_profile.is_shifted else None
        ),
        client_context_uniforms=client_context_uniforms,
    )
    return LinearBanditTrial(
        library=library,
        client_context_profile=client_context_profile,
        client_context_diagnostics=client_context_diagnostics,
        theta_global=theta_global,
        delta_local=delta_local,
        context_ids=context_ids,
        reward_noise=reward_noise,
        exploration_uniforms=exploration_uniforms,
        exploration_actions=exploration_actions,
        client_context_uniforms=client_context_uniforms,
        seed=int(seed),
        heterogeneity=float(heterogeneity),
        digest=digest,
    )


def _normalised_vector(vector: np.ndarray, norm: float) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0.0:
        vector = np.zeros_like(vector)
        vector[0] = norm
        return vector
    return vector * (norm / magnitude)


def _centred_local_effects(raw: np.ndarray, rms_norm: float) -> np.ndarray:
    if rms_norm == 0.0:
        return np.zeros_like(raw)
    centred = raw - np.mean(raw, axis=0, keepdims=True)
    current_rms = float(np.sqrt(np.mean(np.sum(centred * centred, axis=1))))
    if current_rms == 0.0:
        centred[0, 0] = 1.0
        centred -= np.mean(centred, axis=0, keepdims=True)
        current_rms = float(np.sqrt(np.mean(np.sum(centred * centred, axis=1))))
    return centred * (rms_norm / current_rms)


def _sample_client_context_ids(
    sampling_probabilities: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:


    probabilities = np.asarray(sampling_probabilities, dtype=float)
    uniforms = np.asarray(uniforms, dtype=float)
    if probabilities.ndim != 2 or uniforms.ndim != 2:
        raise ValueError("client probabilities and uniforms must both be rank two")
    if probabilities.shape[0] != uniforms.shape[0]:
        raise ValueError("client probabilities and uniforms must share a client axis")
    cdf = np.cumsum(probabilities, axis=1)
    result = np.empty(uniforms.shape, dtype=np.int64)
    for client in range(probabilities.shape[0]):
        result[client] = np.searchsorted(cdf[client], uniforms[client], side="right")
    if np.any(result < 0) or np.any(result >= probabilities.shape[1]):
        raise RuntimeError("inverse-CDF context sampler emitted an invalid row index")
    return result


def _trial_digest(
    library_digest: str,
    *arrays: np.ndarray,
    context_profile_digest: str | None = None,
    client_context_uniforms: np.ndarray | None = None,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(library_digest.encode("ascii"))
    if context_profile_digest is not None:
        hasher.update(context_profile_digest.encode("ascii"))
    for array in arrays:
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    if client_context_uniforms is not None:
        hasher.update(np.ascontiguousarray(client_context_uniforms).view(np.uint8))
    return hasher.hexdigest()
