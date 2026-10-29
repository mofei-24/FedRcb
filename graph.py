from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaticGraph:


    topology: str
    mixing_matrix: np.ndarray
    undirected_edge_count: int
    consensus_slem: float

    @property
    def num_clients(self) -> int:
        return int(self.mixing_matrix.shape[0])

    @property
    def directed_edge_count(self) -> int:
        return 2 * self.undirected_edge_count

    def mix(self, values: np.ndarray, rounds: int) -> np.ndarray:


        if values.shape[0] != self.num_clients:
            raise ValueError("the first data axis must index graph clients")
        if rounds < 0:
            raise ValueError("rounds must be non-negative")


        power = np.linalg.matrix_power(self.mixing_matrix, rounds)
        return np.tensordot(power, np.asarray(values, dtype=float), axes=(1, 0))


def make_static_graph(topology: str, num_clients: int) -> StaticGraph:


    if num_clients < 2:
        raise ValueError("a graph needs at least two clients")
    if topology == "complete":
        matrix = np.full((num_clients, num_clients), 1.0 / num_clients)
        edges = num_clients * (num_clients - 1) // 2
    elif topology == "ring":
        matrix, edges = _ring_mixing_matrix(num_clients)
    elif topology == "star":
        matrix, edges = _star_mixing_matrix(num_clients)
    else:
        raise ValueError("topology must be 'ring', 'complete', or 'star'")
    _validate_mixing_matrix(matrix)
    eigenvalues = np.linalg.eigvalsh(matrix)
    slem = float(np.max(np.abs(eigenvalues[:-1])))
    matrix.setflags(write=False)
    return StaticGraph(
        topology=topology,
        mixing_matrix=matrix,
        undirected_edge_count=edges,
        consensus_slem=slem,
    )


def _ring_mixing_matrix(num_clients: int) -> tuple[np.ndarray, int]:
    if num_clients == 2:
        return np.full((2, 2), 0.5), 1
    matrix = np.zeros((num_clients, num_clients))
    for client in range(num_clients):
        left = (client - 1) % num_clients
        right = (client + 1) % num_clients
        matrix[client, left] = 1.0 / 3.0
        matrix[client, right] = 1.0 / 3.0
        matrix[client, client] = 1.0 / 3.0
    return matrix, num_clients


def _star_mixing_matrix(num_clients: int) -> tuple[np.ndarray, int]:


    if num_clients == 2:
        return np.full((2, 2), 0.5), 1
    matrix = np.zeros((num_clients, num_clients))
    edge_weight = 1.0 / num_clients
    center = 0
    for leaf in range(1, num_clients):
        matrix[center, leaf] = edge_weight
        matrix[leaf, center] = edge_weight
    matrix[center, center] = 1.0 - (num_clients - 1) * edge_weight
    for leaf in range(1, num_clients):
        matrix[leaf, leaf] = 1.0 - edge_weight
    return matrix, num_clients - 1


def _validate_mixing_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("mixing matrix must be square")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("mixing matrix must be symmetric")
    if np.any(matrix < -1e-14):
        raise ValueError("mixing matrix must be non-negative")
    ones = np.ones(matrix.shape[0])
    if not np.allclose(matrix @ ones, ones, atol=1e-12):
        raise ValueError("mixing matrix rows must sum to one")
    if not np.allclose(ones @ matrix, ones, atol=1e-12):
        raise ValueError("mixing matrix columns must sum to one")
