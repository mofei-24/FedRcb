from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from .confidence import theory_ucb_radius


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "synthetic"
    bundle_path: str | None = None
    num_clients: int = 10
    num_actions: int = 10
    global_dim: int = 5
    local_dim: int = 5
    horizon: int = 20_000
    noise_std: float = 0.1
    epoch_length: int = 100
    checkpoint_interval: int = 100
    library_size: int = 100
    context_norm: float = 1.0
    min_context_eigenvalue: float = 0.02
    global_theta_norm: float = 0.8
    client_context_concentration: float = 1.0
    client_context_profile_seed: int = 726_301
    local_energy_fraction: float = 0.75
    regularization: float = 1.0
    forced_exploration: float = 0.1
    forced_exploration_decay_power: float = 1.0
    ucb_confidence_delta: float = 0.1
    ucb_parameter_norm_bound: float = 1.0
    ucb_noise_proxy: float = 0.1
    reward_clip: float = 2.0
    topology: str = "ring"
    gossip_rounds: int = 15
    bit_width: int = 16
    threshold_factor: float = 1.0
    codec_seed: int = 90_271
    cache_tolerance: float = 1e-8
    normalization_tolerance: float = 1e-8
    heterogeneity_values: tuple[float, ...] = (0.0, 0.1, 0.3)
    evaluation_seed_base: int = 20_269_401
    evaluation_seed_count: int = 20
    final_window_rounds: int = 2_000
    bootstrap_replicas: int = 2_000
    bootstrap_confidence: float = 0.95
    parallel_workers: int = 1
    output_dir: str = "results/main"

    @property
    def evaluation_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.evaluation_seed_base, self.evaluation_seed_base + self.evaluation_seed_count))

    @property
    def local_feature_dim(self) -> int:
        return self.global_dim + self.local_dim

    @property
    def joint_feature_dim(self) -> int:
        return self.global_dim + self.num_clients * self.local_dim

    @property
    def epoch_count(self) -> int:
        return (self.horizon + self.epoch_length - 1) // self.epoch_length

    @property
    def context_mode(self) -> str:
        return "movielens" if self.dataset == "movielens" else "shared"

    @property
    def movielens_env_path(self) -> str | None:
        return self.bundle_path

    def effective_forced_exploration(self, time_index: int) -> float:
        epoch = time_index // self.epoch_length + 1
        return self.forced_exploration / float(epoch) ** float(self.forced_exploration_decay_power)

    def confidence_radius(self, completed_pulls: int) -> float:
        if isinstance(completed_pulls, bool) or not isinstance(completed_pulls, int) or completed_pulls < 0:
            raise ValueError("completed_pulls must be a non-negative integer")
        return theory_ucb_radius(
            self.num_clients * completed_pulls,
            noise_proxy=self.ucb_noise_proxy,
            dimension=self.local_feature_dim,
            feature_norm_bound=self.context_norm,
            regularization=self.regularization,
            delta=self.ucb_confidence_delta,
            parameter_norm_bound=self.ucb_parameter_norm_bound,
        )

    def with_overrides(self, **changes: object) -> "ExperimentConfig":
        return replace(self, **changes)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        positive_integers = (
            "num_clients", "num_actions", "global_dim", "local_dim", "horizon",
            "epoch_length", "checkpoint_interval", "library_size", "gossip_rounds",
            "bit_width", "evaluation_seed_count", "final_window_rounds",
            "bootstrap_replicas", "parallel_workers",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("client_context_profile_seed", "codec_seed", "evaluation_seed_base"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        positive_numbers = (
            "context_norm", "global_theta_norm", "client_context_concentration",
            "regularization", "ucb_parameter_norm_bound", "ucb_noise_proxy",
            "reward_clip", "forced_exploration_decay_power",
        )
        for name in positive_numbers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("noise_std", "min_context_eigenvalue", "cache_tolerance", "normalization_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("ucb_confidence_delta", "bootstrap_confidence", "local_energy_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly in (0, 1)")
        for name in ("forced_exploration", "threshold_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.dataset not in {"synthetic", "movielens", "jester"}:
            raise ValueError("dataset must be synthetic, movielens, or jester")
        if self.dataset != "synthetic" and not self.bundle_path:
            raise ValueError("real-data experiments require bundle_path")
        if self.num_clients < 2 or self.num_actions < 2:
            raise ValueError("at least two clients and two actions are required")
        if self.library_size < self.num_actions:
            raise ValueError("library_size must be at least num_actions")
        if self.topology not in {"ring", "complete", "star"}:
            raise ValueError("topology must be ring, complete, or star")
        if not 3 <= self.bit_width <= 24:
            raise ValueError("bit_width must lie in [3, 24]")
        if self.epoch_length > self.horizon:
            raise ValueError("epoch_length cannot exceed horizon")
        if self.final_window_rounds > self.horizon:
            raise ValueError("final_window_rounds cannot exceed horizon")
        if self.horizon % self.checkpoint_interval or self.final_window_rounds % self.checkpoint_interval:
            raise ValueError("horizon and final_window_rounds must be checkpoint multiples")
        if not self.heterogeneity_values or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in self.heterogeneity_values
        ):
            raise ValueError("heterogeneity_values must contain finite non-negative numbers")
