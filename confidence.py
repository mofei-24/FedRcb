from __future__ import annotations

import math
from numbers import Integral, Real


def _positive_finite(name: str, value: Real) -> float:


    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return numeric


def _validate_dimension(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("dimension must be an integer")
    dimension = int(value)
    if dimension <= 0:
        raise ValueError("dimension must be strictly positive")
    return dimension


def _validate_sample_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("t must be a non-negative integer")
    sample_count = int(value)
    if sample_count < 0:
        raise ValueError("t must be non-negative")
    return sample_count


def theory_ucb_radius(
    t: int,
    *,
    noise_proxy: Real,
    dimension: int,
    feature_norm_bound: Real,
    regularization: Real,
    delta: Real,
    parameter_norm_bound: Real,
) -> float:


    sample_count = _validate_sample_count(t)
    sigma = _positive_finite("noise_proxy", noise_proxy)
    d = _validate_dimension(dimension)
    feature_bound = _positive_finite("feature_norm_bound", feature_norm_bound)
    ridge = _positive_finite("regularization", regularization)
    confidence = _positive_finite("delta", delta)
    if confidence >= 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    parameter_bound = _positive_finite("parameter_norm_bound", parameter_norm_bound)

    log_term = d * math.log(1.0 + sample_count * feature_bound**2 / ridge)
    confidence_term = 2.0 * math.log(1.0 / confidence)
    return sigma * math.sqrt(log_term + confidence_term) + math.sqrt(ridge) * parameter_bound
