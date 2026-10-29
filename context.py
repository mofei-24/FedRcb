from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class ContextDiagnostics:


    global_min_eigenvalue: float
    local_min_eigenvalue: float
    full_min_eigenvalue: float

    def as_dict(self) -> dict[str, float]:
        return {
            "global_min_eigenvalue": self.global_min_eigenvalue,
            "local_min_eigenvalue": self.local_min_eigenvalue,
            "full_min_eigenvalue": self.full_min_eigenvalue,
        }


@dataclass(frozen=True)
class PublicClientContextProfile:


    base_probabilities: np.ndarray
    preference_probabilities: np.ndarray
    sampling_probabilities: np.ndarray
    mixture_eta: float
    profile_strength: float
    profile_seed: int
    digest: str
    kind: str = "structured"
    concentration: float | None = None

    @property
    def num_clients(self) -> int:
        return int(self.sampling_probabilities.shape[0])

    @property
    def library_size(self) -> int:
        return int(self.sampling_probabilities.shape[1])

    @property
    def is_shifted(self) -> bool:
        return self.kind == "dirichlet" or self.mixture_eta > 0.0


@dataclass(frozen=True)
class ClientContextDiagnostics:


    per_client: tuple[ContextDiagnostics, ...]
    total_variation_from_base: np.ndarray

    @property
    def global_min_eigenvalue(self) -> float:
        return min(item.global_min_eigenvalue for item in self.per_client)

    @property
    def local_min_eigenvalue(self) -> float:
        return min(item.local_min_eigenvalue for item in self.per_client)

    @property
    def full_min_eigenvalue(self) -> float:
        return min(item.full_min_eigenvalue for item in self.per_client)

    def as_dict(self) -> dict[str, float]:
        return {
            "global_min_eigenvalue": self.global_min_eigenvalue,
            "local_min_eigenvalue": self.local_min_eigenvalue,
            "full_min_eigenvalue": self.full_min_eigenvalue,
            "client_total_variation_mean": float(
                np.mean(self.total_variation_from_base)
            ),
            "client_total_variation_max": float(
                np.max(self.total_variation_from_base)
            ),
        }


@dataclass(frozen=True)
class PublicContextLibrary:


    x_global: np.ndarray
    x_local: np.ndarray
    context_norm: float
    sampling_law: str
    sampling_probabilities: np.ndarray
    diagnostics: ContextDiagnostics
    digest: str


    local_energy_fraction: float | None = None

    @property
    def library_size(self) -> int:
        return int(self.x_global.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.x_global.shape[1])

    @property
    def global_dim(self) -> int:
        return int(self.x_global.shape[2])

    @property
    def local_dim(self) -> int:
        return int(self.x_local.shape[2])

    @property
    def is_uniform(self) -> bool:
        return self.sampling_law == "uniform"

    def slate(self, context_id: int) -> tuple[np.ndarray, np.ndarray]:


        if not 0 <= context_id < self.library_size:
            raise IndexError("context_id lies outside the public library")
        return self.x_global[context_id], self.x_local[context_id]


def generate_public_context_library(
    *,
    library_size: int,
    num_actions: int,
    global_dim: int,
    local_dim: int,
    context_norm: float,
    min_eigenvalue: float,
    rng: np.random.Generator,
    required_client_sampling_probabilities: np.ndarray | None = None,
    local_energy_fraction: float | None = None,
    max_attempts: int = 100,
) -> PublicContextLibrary:


    if library_size <= 0 or num_actions <= 0:
        raise ValueError("library_size and num_actions must be positive")
    if global_dim <= 0 or local_dim <= 0:
        raise ValueError("global_dim and local_dim must be positive")
    if context_norm <= 0:
        raise ValueError("context_norm must be positive")
    if min_eigenvalue < 0:
        raise ValueError("min_eigenvalue must be non-negative")
    if local_energy_fraction is not None and not (
        0.0 < float(local_energy_fraction) < 1.0
    ):
        raise ValueError("local_energy_fraction must lie strictly in (0, 1)")
    probabilities = context_sampling_probabilities(library_size)
    required_client_probabilities = _validate_client_sampling_probabilities(
        required_client_sampling_probabilities,
        library_size=library_size,
    )

    full_dim = global_dim + local_dim
    for _ in range(max_attempts):
        raw = rng.normal(size=(library_size, num_actions, full_dim))
        if local_energy_fraction is None:
            norm = np.linalg.norm(raw, axis=-1, keepdims=True)
            features = context_norm * raw / np.maximum(norm, np.finfo(float).tiny)
        else:
            fraction = float(local_energy_fraction)
            raw_global = raw[..., :global_dim]
            raw_local = raw[..., global_dim:]
            norm_global = np.linalg.norm(raw_global, axis=-1, keepdims=True)
            norm_local = np.linalg.norm(raw_local, axis=-1, keepdims=True)
            tiny = np.finfo(float).tiny
            x_global = (
                context_norm
                * math.sqrt(1.0 - fraction)
                * raw_global
                / np.maximum(norm_global, tiny)
            )
            x_local = (
                context_norm
                * math.sqrt(fraction)
                * raw_local
                / np.maximum(norm_local, tiny)
            )
            features = np.concatenate((x_global, x_local), axis=-1)
        x_global = features[..., :global_dim]
        x_local = features[..., global_dim:]
        diagnostics = context_diagnostics(
            x_global,
            x_local,
            sampling_probabilities=probabilities,
        )
        client_diagnostics = (
            None
            if required_client_probabilities is None
            else context_diagnostics_by_client(
                x_global,
                x_local,
                sampling_probabilities=required_client_probabilities,
                base_probabilities=probabilities,
            )
        )
        if (
            diagnostics.global_min_eigenvalue >= min_eigenvalue
            and diagnostics.local_min_eigenvalue >= min_eigenvalue
            and (
                client_diagnostics is None
                or (
                    client_diagnostics.global_min_eigenvalue >= min_eigenvalue
                    and client_diagnostics.local_min_eigenvalue >= min_eigenvalue
                    and client_diagnostics.full_min_eigenvalue >= min_eigenvalue
                )
            )
        ):
            _freeze(x_global)
            _freeze(x_local)
            _freeze(probabilities)
            digest = _array_digest(x_global, x_local, probabilities)
            return PublicContextLibrary(
                x_global=x_global,
                x_local=x_local,
                context_norm=float(context_norm),
                sampling_law="uniform",
                sampling_probabilities=probabilities,
                diagnostics=diagnostics,
                digest=digest,
                local_energy_fraction=(
                    None if local_energy_fraction is None else float(local_energy_fraction)
                ),
            )
    raise RuntimeError(
        "could not generate a public context library satisfying the requested "
        f"minimum eigenvalue {min_eigenvalue} after {max_attempts} attempts"
    )


def generate_public_client_context_profile(
    *,
    num_clients: int,
    library_size: int,
    profile_seed: int,
    concentration: float,
    base_probabilities: np.ndarray,
) -> PublicClientContextProfile:
    if num_clients <= 0 or library_size <= 0:
        raise ValueError("num_clients and library_size must be positive")
    if profile_seed < 0:
        raise ValueError("profile_seed must be non-negative")
    if isinstance(concentration, bool) or not isinstance(concentration, (int, float)):
        raise ValueError("concentration must be a positive number")
    if not math.isfinite(float(concentration)) or concentration <= 0:
        raise ValueError("concentration must be finite and positive")
    base = np.array(
        _validate_row_probabilities(base_probabilities, library_size=library_size),
        dtype=float,
        copy=True,
    )
    rng = np.random.default_rng(profile_seed)
    probabilities = np.asarray(
        rng.dirichlet(np.full(library_size, float(concentration)), size=num_clients),
        dtype=float,
    )
    preferences = probabilities.copy()
    _validate_client_sampling_probabilities(probabilities, library_size=library_size)
    for array in (base, preferences, probabilities):
        _freeze(array)
    return PublicClientContextProfile(
        base_probabilities=base,
        preference_probabilities=preferences,
        sampling_probabilities=probabilities,
        mixture_eta=1.0,
        profile_strength=0.9,
        profile_seed=int(profile_seed),
        digest=_profile_digest(
            mixture_eta=1.0,
            profile_strength=0.9,
            profile_seed=profile_seed,
            base_probabilities=base,
            preference_probabilities=preferences,
            sampling_probabilities=probabilities,
            kind="dirichlet",
            concentration=float(concentration),
        ),
        kind="dirichlet",
        concentration=float(concentration),
    )


def context_sampling_probabilities(library_size: int) -> np.ndarray:
    if library_size <= 0:
        raise ValueError("library_size must be positive")
    weights = np.ones(library_size, dtype=float)
    return weights / np.sum(weights)


def context_diagnostics(
    x_global: np.ndarray,
    x_local: np.ndarray,
    *,
    sampling_probabilities: np.ndarray | None = None,
) -> ContextDiagnostics:


    x_global = np.asarray(x_global, dtype=float)
    x_local = np.asarray(x_local, dtype=float)
    if x_global.ndim != 3 or x_local.ndim != 3:
        raise ValueError("context arrays must have shape [library, action, feature]")
    if x_global.shape[:2] != x_local.shape[:2]:
        raise ValueError("global/local context arrays must share library and action axes")
    probabilities = _validate_row_probabilities(
        sampling_probabilities,
        library_size=x_global.shape[0],
    )
    global_gram = _expected_gram(x_global, probabilities)
    local_gram = _expected_gram(x_local, probabilities)
    full_gram = _expected_gram(np.concatenate((x_global, x_local), axis=-1), probabilities)
    return ContextDiagnostics(
        global_min_eigenvalue=float(np.linalg.eigvalsh(global_gram)[0]),
        local_min_eigenvalue=float(np.linalg.eigvalsh(local_gram)[0]),
        full_min_eigenvalue=float(np.linalg.eigvalsh(full_gram)[0]),
    )


def context_diagnostics_by_client(
    x_global: np.ndarray,
    x_local: np.ndarray,
    *,
    sampling_probabilities: np.ndarray,
    base_probabilities: np.ndarray | None = None,
) -> ClientContextDiagnostics:


    x_global = np.asarray(x_global, dtype=float)
    x_local = np.asarray(x_local, dtype=float)
    if x_global.ndim != 3 or x_local.ndim != 3:
        raise ValueError("context arrays must have shape [library, action, feature]")
    if x_global.shape[:2] != x_local.shape[:2]:
        raise ValueError("global/local context arrays must share library and action axes")
    probabilities = _validate_client_sampling_probabilities(
        sampling_probabilities,
        library_size=x_global.shape[0],
    )
    if base_probabilities is None:
        base = np.full(x_global.shape[0], 1.0 / x_global.shape[0])
    else:
        base = _validate_row_probabilities(
            base_probabilities,
            library_size=x_global.shape[0],
        )
    per_client = tuple(
        context_diagnostics(
            x_global,
            x_local,
            sampling_probabilities=probabilities[client],
        )
        for client in range(probabilities.shape[0])
    )
    variation = 0.5 * np.sum(np.abs(probabilities - base[None, :]), axis=1)
    _freeze(variation)
    return ClientContextDiagnostics(
        per_client=per_client,
        total_variation_from_base=variation,
    )


def expected_feature_gram(
    features: np.ndarray, *, sampling_probabilities: np.ndarray
) -> np.ndarray:


    features = np.asarray(features, dtype=float)
    if features.ndim != 3:
        raise ValueError("features must have shape [library, action, feature]")
    probabilities = _validate_row_probabilities(
        sampling_probabilities,
        library_size=features.shape[0],
    )
    return _expected_gram(features, probabilities)


def _validate_row_probabilities(
    probabilities: np.ndarray | None, *, library_size: int
) -> np.ndarray:
    if probabilities is None:
        return np.full(library_size, 1.0 / library_size)
    result = np.asarray(probabilities, dtype=float)
    if result.shape != (library_size,):
        raise ValueError("sampling probabilities must have one entry per catalogue row")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("sampling probabilities must be finite and strictly positive")
    if not np.isclose(np.sum(result), 1.0, atol=1e-12):
        raise ValueError("sampling probabilities must sum to one")
    return result


def _validate_client_sampling_probabilities(
    probabilities: np.ndarray | None, *, library_size: int
) -> np.ndarray | None:
    if probabilities is None:
        return None
    result = np.asarray(probabilities, dtype=float)
    if result.ndim != 2 or result.shape[1] != library_size:
        raise ValueError(
            "client sampling probabilities must have shape [client, catalogue row]"
        )
    if result.shape[0] <= 0:
        raise ValueError("at least one client sampling distribution is required")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(
            "client sampling probabilities must be finite and strictly positive"
        )
    if not np.allclose(np.sum(result, axis=1), 1.0, atol=1e-12):
        raise ValueError("each client sampling distribution must sum to one")
    return result


def _expected_gram(features: np.ndarray, row_probabilities: np.ndarray) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    action_count = features.shape[1]
    weights = np.repeat(row_probabilities / action_count, action_count)
    return flat.T @ (flat * weights[:, None])


def _array_digest(*arrays: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    return hasher.hexdigest()


def _profile_digest(
    *,
    mixture_eta: float,
    profile_strength: float,
    profile_seed: int,
    base_probabilities: np.ndarray,
    preference_probabilities: np.ndarray,
    sampling_probabilities: np.ndarray,
    kind: str = "structured",
    concentration: float | None = None,
) -> str:
    hasher = hashlib.sha256()
    concentration_text = "None" if concentration is None else f"{float(concentration):.17g}"
    hasher.update(
        (
            f"kind={kind};eta={float(mixture_eta):.17g};"
            f"strength={float(profile_strength):.17g};"
            f"concentration={concentration_text};"
            f"seed={int(profile_seed)}"
        ).encode("ascii")
    )
    for array in (base_probabilities, preference_probabilities, sampling_probabilities):
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    return hasher.hexdigest()


def _freeze(array: np.ndarray) -> None:
    array.setflags(write=False)
