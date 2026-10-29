from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrozenRidgeState:


    theta: np.ndarray
    inverse_design: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.theta.shape[0])

    def ucb_scores(self, features: np.ndarray, alpha: float) -> np.ndarray:


        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError("features must have shape [action, state dimension]")
        mean = values @ self.theta
        variance = np.einsum("ki,ij,kj->k", values, self.inverse_design, values)
        return mean + alpha * np.sqrt(np.maximum(variance, 0.0))


def concatenate_features(x_global: np.ndarray, x_local: np.ndarray) -> np.ndarray:


    x_global = np.asarray(x_global, dtype=float)
    x_local = np.asarray(x_local, dtype=float)
    if x_global.ndim != 2 or x_local.ndim != 2 or x_global.shape[0] != x_local.shape[0]:
        raise ValueError("feature slates must be matrices with the same action count")
    return np.concatenate((x_global, x_local), axis=1)


def reduced_payload_dimension(global_dim: int) -> int:


    if global_dim <= 0:
        raise ValueError("global_dim must be positive")
    return global_dim * (global_dim + 1) // 2 + global_dim


def pack_symmetric(matrix: np.ndarray) -> np.ndarray:


    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(values, values.T, atol=1e-10):
        raise ValueError("matrix must be symmetric before packing")
    indices = np.triu_indices(values.shape[0])
    return values[indices].copy()


def unpack_symmetric(values: np.ndarray, dimension: int) -> np.ndarray:


    packed = np.asarray(values, dtype=float)
    expected = dimension * (dimension + 1) // 2
    if packed.shape != (expected,):
        raise ValueError(
            f"packed symmetric vector has shape {packed.shape}, expected ({expected},)"
        )
    result = np.zeros((dimension, dimension), dtype=float)
    indices = np.triu_indices(dimension)
    result[indices] = packed
    result[(indices[1], indices[0])] = packed
    return result


def initialise_local_moments(
    num_clients: int, global_dim: int, local_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


    return (
        np.zeros((num_clients, global_dim, global_dim), dtype=float),
        np.zeros((num_clients, global_dim, local_dim), dtype=float),
        np.zeros((num_clients, local_dim, local_dim), dtype=float),
        np.zeros((num_clients, global_dim), dtype=float),
        np.zeros((num_clients, local_dim), dtype=float),
    )


def add_local_moment(
    global_design: np.ndarray,
    cross_design: np.ndarray,
    local_design: np.ndarray,
    global_response: np.ndarray,
    local_response: np.ndarray,
    *,
    client: int,
    x_global: np.ndarray,
    x_local: np.ndarray,
    reward: float,
) -> None:


    xg = np.asarray(x_global, dtype=float)
    xl = np.asarray(x_local, dtype=float)
    global_design[client] += np.outer(xg, xg)
    cross_design[client] += np.outer(xg, xl)
    local_design[client] += np.outer(xl, xl)
    global_response[client] += xg * float(reward)
    local_response[client] += xl * float(reward)


def local_reduced_statistics(
    global_design: np.ndarray,
    cross_design: np.ndarray,
    local_design: np.ndarray,
    global_response: np.ndarray,
    local_response: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:


    if regularization <= 0:
        raise ValueError("regularization must be positive")
    clients, global_dim, local_dim = cross_design.shape
    if global_design.shape != (clients, global_dim, global_dim):
        raise ValueError("global_design shape does not match cross_design")
    if local_design.shape != (clients, local_dim, local_dim):
        raise ValueError("local_design shape does not match cross_design")
    if global_response.shape != (clients, global_dim):
        raise ValueError("global_response shape does not match cross_design")
    if local_response.shape != (clients, local_dim):
        raise ValueError("local_response shape does not match cross_design")

    reduced_design = np.empty_like(global_design)
    reduced_response = np.empty_like(global_response)
    identity = np.eye(local_dim)
    for client in range(clients):
        local_ridge = local_design[client] + regularization * identity
        inverse_times_cross_transpose = np.linalg.solve(
            local_ridge, cross_design[client].T
        )
        inverse_times_response = np.linalg.solve(local_ridge, local_response[client])
        design = global_design[client] - cross_design[client] @ inverse_times_cross_transpose
        reduced_design[client] = 0.5 * (design + design.T)
        reduced_response[client] = (
            global_response[client] - cross_design[client] @ inverse_times_response
        )
    return reduced_design, reduced_response


def pack_reduced_statistics(
    reduced_design: np.ndarray, reduced_response: np.ndarray
) -> np.ndarray:


    designs = np.asarray(reduced_design, dtype=float)
    responses = np.asarray(reduced_response, dtype=float)
    if designs.ndim != 3 or designs.shape[1] != designs.shape[2]:
        raise ValueError("reduced_design must have shape [client, global_dim, global_dim]")
    if responses.shape != (designs.shape[0], designs.shape[1]):
        raise ValueError("reduced_response shape does not match reduced_design")
    return np.stack(
        [np.concatenate((pack_symmetric(designs[client]), responses[client])) for client in range(designs.shape[0])]
    )


def unpack_reduced_statistics(
    payload: np.ndarray, global_dim: int
) -> tuple[np.ndarray, np.ndarray]:


    vector = np.asarray(payload, dtype=float)
    packed_dim = global_dim * (global_dim + 1) // 2
    if vector.shape != (packed_dim + global_dim,):
        raise ValueError("payload dimension does not match global_dim")
    return unpack_symmetric(vector[:packed_dim], global_dim), vector[packed_dim:].copy()


def normalize_reduced_payload(
    payload: np.ndarray, *, global_dim: int, completed_pulls: int, reward_clip: float
) -> np.ndarray:


    if completed_pulls <= 0:
        raise ValueError("completed_pulls must be positive")
    if reward_clip <= 0:
        raise ValueError("reward_clip must be positive")
    vector = np.asarray(payload, dtype=float)
    symmetric_dim = global_dim * (global_dim + 1) // 2
    expected = symmetric_dim + global_dim
    if vector.shape != (expected,):
        raise ValueError("payload dimension does not match global_dim")
    result = vector.copy()
    result[:symmetric_dim] /= float(completed_pulls)
    result[symmetric_dim:] /= float(completed_pulls * reward_clip)
    return result


def denormalize_reduced_payload(
    normalized: np.ndarray,
    *,
    global_dim: int,
    completed_pulls: int,
    reward_clip: float,
) -> np.ndarray:


    if completed_pulls <= 0:
        raise ValueError("completed_pulls must be positive")
    if reward_clip <= 0:
        raise ValueError("reward_clip must be positive")
    vector = np.asarray(normalized, dtype=float).copy()
    symmetric_dim = global_dim * (global_dim + 1) // 2
    expected = symmetric_dim + global_dim
    if vector.shape != (expected,):
        raise ValueError("payload dimension does not match global_dim")
    vector[:symmetric_dim] *= float(completed_pulls)
    vector[symmetric_dim:] *= float(completed_pulls * reward_clip)
    return vector


def project_symmetric_psd(matrix: np.ndarray) -> np.ndarray:


    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected_values = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * projected_values) @ eigenvectors.T
    projected = 0.5 * (projected + projected.T)
    return projected


def client_state_from_reduced_statistics(
    global_reduced_design: np.ndarray,
    global_reduced_response: np.ndarray,
    *,
    local_cross_design: np.ndarray,
    local_design: np.ndarray,
    local_response: np.ndarray,
    regularization: float,
) -> FrozenRidgeState:


    reduced_design = np.asarray(global_reduced_design, dtype=float)
    reduced_response = np.asarray(global_reduced_response, dtype=float)
    cross = np.asarray(local_cross_design, dtype=float)
    local = np.asarray(local_design, dtype=float)
    local_b = np.asarray(local_response, dtype=float)
    global_dim, local_dim = cross.shape
    if reduced_design.shape != (global_dim, global_dim):
        raise ValueError("global reduced design has incompatible shape")
    if reduced_response.shape != (global_dim,):
        raise ValueError("global reduced response has incompatible shape")
    if local.shape != (local_dim, local_dim) or local_b.shape != (local_dim,):
        raise ValueError("local moment shapes do not match cross design")
    if regularization <= 0:
        raise ValueError("regularization must be positive")

    design = project_symmetric_psd(reduced_design)
    global_ridge = design + regularization * np.eye(global_dim)
    local_ridge = local + regularization * np.eye(local_dim)
    global_inverse = np.linalg.inv(global_ridge)
    local_inverse = np.linalg.inv(local_ridge)
    theta_global = global_inverse @ reduced_response
    theta_local = local_inverse @ (local_b - cross.T @ theta_global)
    cross_local_inverse = cross @ local_inverse
    top_right = -global_inverse @ cross_local_inverse
    bottom_left = top_right.T
    bottom_right = (
        local_inverse + local_inverse @ cross.T @ global_inverse @ cross @ local_inverse
    )
    inverse_design = np.block(
        [[global_inverse, top_right], [bottom_left, bottom_right]]
    )
    inverse_design = 0.5 * (inverse_design + inverse_design.T)
    state = FrozenRidgeState(
        theta=np.concatenate((theta_global, theta_local)), inverse_design=inverse_design
    )
    return state
