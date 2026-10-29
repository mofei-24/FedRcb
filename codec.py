from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .graph import StaticGraph


def directed_edges_from_mixing(mixing_matrix: np.ndarray) -> tuple[tuple[int, int], ...]:


    matrix = np.asarray(mixing_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("mixing_matrix must be square")
    return tuple(
        (receiver, sender)
        for receiver in range(matrix.shape[0])
        for sender in range(matrix.shape[1])
        if receiver != sender and matrix[receiver, sender] > 0.0
    )


_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class ResidualScaleSchedule:


    bit_width: int
    rounds: int

    def __post_init__(self) -> None:
        if isinstance(self.bit_width, bool) or not isinstance(self.bit_width, int):
            raise ValueError("bit_width must be an integer")
        if self.bit_width < 3 or self.bit_width > 24:
            raise ValueError("safe residual transport requires bit_width in [3, 24]")
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")

    @property
    def levels(self) -> int:
        return (1 << self.bit_width) - 1

    @property
    def state_bounds(self) -> np.ndarray:
        multiplier = 1.0 + 4.0 / float(self.levels - 2)
        result = multiplier ** np.arange(self.rounds + 1, dtype=float)
        result.setflags(write=False)
        return result

    @property
    def amplitudes(self) -> np.ndarray:
        result = 2.0 * self.state_bounds[:-1] * self.levels / float(self.levels - 2)
        result.setflags(write=False)
        return result

    @property
    def errors(self) -> np.ndarray:
        result = 2.0 * self.amplitudes / float(self.levels)
        result.setflags(write=False)
        return result

    def amplitude(self, stage: int) -> float:
        return float(self.amplitudes[_stage_index(stage, self.rounds)])

    def error_bound(self, stage: int) -> float:
        return float(self.errors[_stage_index(stage, self.rounds)])

    def state_bound(self, stage: int) -> float:
        if stage < 0 or stage > self.rounds:
            raise ValueError("stage lies outside public residual schedule")
        return float(self.state_bounds[stage])


class CounterStochasticResidualCodec:


    def __init__(self, bit_width: int, seed: int, *, tolerance: float = 1e-8) -> None:
        if isinstance(bit_width, bool) or not isinstance(bit_width, int):
            raise ValueError("bit_width must be an integer")
        if bit_width < 1 or bit_width > 24:
            raise ValueError("bit_width must be in [1, 24]")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.bit_width = int(bit_width)
        self.seed = int(seed) & _UINT64_MASK
        self.levels = (1 << self.bit_width) - 1
        self.tolerance = float(tolerance)

    def encode(
        self,
        values: np.ndarray,
        *,
        amplitude: float,
        epoch: int,
        stage: int,
    ) -> np.ndarray:


        raw = np.asarray(values, dtype=float)
        if raw.ndim != 2:
            raise ValueError("residual values must have shape [sender, coordinate]")
        if not np.all(np.isfinite(raw)):
            raise ValueError("residual codec input must be finite")
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")
        normalized = raw / float(amplitude)
        if np.any(np.abs(normalized) > 1.0 + self.tolerance):
            raise ValueError("residual exceeded public codec range; refusing saturation")
        normalized = np.clip(normalized, -1.0, 1.0)
        position = (normalized + 1.0) * (self.levels / 2.0)
        lower = np.floor(position).astype(np.uint32)
        upper = np.minimum(lower + 1, self.levels).astype(np.uint32)
        probability_upper = position - lower
        draws = self._uniform_grid(raw.shape, epoch=epoch, stage=stage)
        return np.where(draws < probability_upper, upper, lower).astype(np.uint32)

    def decode(self, codes: np.ndarray, *, amplitude: float) -> np.ndarray:
        encoded = np.asarray(codes)
        if np.any(encoded < 0) or np.any(encoded > self.levels):
            raise ValueError("residual code lies outside fixed-width range")
        return float(amplitude) * (
            -1.0 + 2.0 * np.asarray(encoded, dtype=float) / float(self.levels)
        )

    def _uniform_grid(self, shape: tuple[int, int], *, epoch: int, stage: int) -> np.ndarray:
        senders = np.arange(shape[0], dtype=np.uint64)[:, None]
        coordinates = np.arange(shape[1], dtype=np.uint64)[None, :]


        with np.errstate(over="ignore"):
            counter = (
                np.uint64(self.seed)
                ^ (np.uint64(int(epoch) & _UINT64_MASK) * np.uint64(0x9E3779B185EBCA87))
                ^ (np.uint64(int(stage) & _UINT64_MASK) * np.uint64(0xC2B2AE3D27D4EB4F))
                ^ (senders * np.uint64(0x165667B19E3779F9))
                ^ (coordinates * np.uint64(0xD6E8FEB86659FD93))
            )
            mixed = counter + np.uint64(0x9E3779B97F4A7C15)
            mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            mixed = mixed ^ (mixed >> np.uint64(31))
        return (mixed >> np.uint64(11)).astype(float) * (1.0 / (1 << 53))


@dataclass
class ResidualPacketLedger:


    bit_width: int
    scalar_micro_packets: int = 0
    vector_frames: int = 0
    payload_bits: int = 0
    flag_bits: int = 0
    residual_code_bits: int = 0
    scheduled_scalar_slots: int = 0
    active_residual_events: int = 0
    range_violation_count: int = 0
    _hasher: object = field(default_factory=hashlib.sha256, repr=False)

    def note_range_violation(self) -> None:
        self.range_violation_count += 1


    def record_quantized(
        self,
        flags: np.ndarray,
        codes: np.ndarray,
        *,
        epoch: int,
        stage: int,
        directed_edges: tuple[tuple[int, int], ...],
    ) -> None:
        active = np.asarray(flags, dtype=bool)
        encoded = np.asarray(codes, dtype=np.uint32)
        if active.shape != encoded.shape or active.ndim != 2:
            raise ValueError("flags and codes must match [sender, coordinate]")
        coordinate_count = int(active.shape[1])
        active_events = 0
        for receiver, sender in directed_edges:
            self._update_prefix(epoch, stage, receiver, sender)
            self._hasher.update(np.asarray(active[sender], dtype=np.uint8).tobytes())

            self._hasher.update(np.asarray(encoded[sender], dtype=">u4").tobytes())
            active_events += int(np.count_nonzero(active[sender]))
        slots = len(directed_edges) * coordinate_count
        self.scalar_micro_packets += slots
        self.scheduled_scalar_slots += slots
        self.vector_frames += len(directed_edges)
        self.active_residual_events += active_events
        self.flag_bits += slots
        code_bits = active_events * self.bit_width
        self.residual_code_bits += code_bits
        self.payload_bits += slots + code_bits

    def _update_prefix(
        self, epoch: int, stage: int, receiver: int, sender: int
    ) -> None:
        prefix = np.asarray((epoch, stage, receiver, sender, 1), dtype=">u4")
        self._hasher.update(prefix.tobytes())

    @property
    def digest(self) -> str:
        return self._hasher.hexdigest()


class ProgressiveResidualTransport:
    def __init__(
        self,
        graph: StaticGraph,
        *,
        rounds: int,
        bit_width: int,
        codec_seed: int,
        threshold_factor: float = 1.0,
        tolerance: float = 1e-8,
    ) -> None:
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        if graph.num_clients < 2:
            raise ValueError("transport graph needs at least two clients")
        if not 0.0 <= threshold_factor <= 1.0:
            raise ValueError("threshold_factor must lie in [0, 1]")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.graph = graph
        self.rounds = int(rounds)
        self.bit_width = bit_width
        self.threshold_factor = float(threshold_factor)
        self.tolerance = float(tolerance)
        self.directed_edges = directed_edges_from_mixing(graph.mixing_matrix)
        if len(self.directed_edges) != graph.directed_edge_count:
            raise RuntimeError("directed edge enumeration disagrees with graph accounting")
        self.schedule = ResidualScaleSchedule(bit_width, self.rounds)
        self.codec = CounterStochasticResidualCodec(bit_width, codec_seed, tolerance=tolerance)
        self.ledger = ResidualPacketLedger(bit_width)
        self._publication: np.ndarray | None = None
        self._cache: np.ndarray | None = None

    @property
    def publication(self) -> np.ndarray:
        if self._publication is None:
            raise RuntimeError("transport has not processed an epoch")
        result = self._publication.copy()
        result.setflags(write=False)
        return result

    @property
    def receiver_cache(self) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("transport has not processed an epoch")
        result = self._cache.copy()
        result.setflags(write=False)
        return result

    def run_epoch(self, values: np.ndarray, *, epoch: int) -> np.ndarray:
        current = np.asarray(values, dtype=float).copy()
        if current.ndim != 2 or current.shape[0] != self.graph.num_clients:
            raise ValueError("values must have shape [graph client, coordinate]")
        if not np.all(np.isfinite(current)):
            raise ValueError("transport values must be finite")
        self._ensure_state(current.shape[1])
        assert self._publication is not None and self._cache is not None
        for stage in range(self.rounds):
            bound = self.schedule.state_bound(stage)
            current_max = float(np.max(np.abs(current))) if current.size else 0.0
            if current_max > bound + self.tolerance:
                self.ledger.note_range_violation()
                raise ValueError(
                    f"stage {stage} state exceeded B_s={bound:g}: {current_max:g}"
                )
            residual = current - self._publication[stage]
            residual_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
            amplitude = self.schedule.amplitude(stage)
            error_bound = self.schedule.error_bound(stage)
            threshold = self.threshold_factor * error_bound
            if residual_abs > amplitude + self.tolerance:
                self.ledger.note_range_violation()
                raise ValueError(
                    f"stage {stage} residual exceeded A_s={amplitude:g}: {residual_abs:g}"
                )
            active = np.abs(residual) > threshold
            codes = self.codec.encode(
                residual, amplitude=amplitude, epoch=epoch, stage=stage
            )
            decoded = self.codec.decode(codes, amplitude=amplitude)
            decoded_delta = np.where(active, decoded, 0.0)
            padded_codes = np.where(active, codes, 0).astype(np.uint32)
            self.ledger.record_quantized(
                active, padded_codes, epoch=epoch, stage=stage + 1,
                directed_edges=self.directed_edges,
            )
            self._publication[stage] += decoded_delta
            for receiver, sender in self.directed_edges:
                self._cache[stage, receiver, sender] += decoded_delta[sender]
            sources = np.broadcast_to(
                self._publication[stage][None, :, :],
                (self.graph.num_clients, self.graph.num_clients, current.shape[1]),
            ).copy()
            for receiver, sender in self.directed_edges:
                sources[receiver, sender] = self._cache[stage, receiver, sender]
            current = np.einsum("rs,rsd->rd", self.graph.mixing_matrix, sources)
        return current

    def _ensure_state(self, dimension: int) -> None:
        shape = (self.rounds, self.graph.num_clients, dimension)
        cache_shape = (self.rounds, self.graph.num_clients, self.graph.num_clients, dimension)
        if self._publication is None:
            self._publication = np.zeros(shape, dtype=float)
            self._cache = np.zeros(cache_shape, dtype=float)
        elif self._publication.shape != shape or self._cache is None or self._cache.shape != cache_shape:
            raise ValueError("persistent transport state dimension changed across epochs")


def _stage_index(stage: int, rounds: int) -> int:
    if stage < 0 or stage >= rounds:
        raise ValueError("stage lies outside public residual schedule")
    return int(stage)

