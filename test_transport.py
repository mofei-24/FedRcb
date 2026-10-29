import numpy as np
import pytest

from fedrcb.codec import CounterStochasticResidualCodec, ProgressiveResidualTransport
from fedrcb.graph import make_static_graph


def test_counter_rounding_is_unbiased_and_rejects_out_of_range_values():
    codec = CounterStochasticResidualCodec(4, 123)
    values = np.full((1, 20_000), 0.17)
    codes = codec.encode(values, amplitude=1.0, epoch=1, stage=0)
    decoded = codec.decode(codes, amplitude=1.0)
    assert abs(decoded.mean() - 0.17) < 0.003
    assert np.max(np.abs(decoded - values)) <= 2 / 15
    assert np.array_equal(codes, codec.encode(values, amplitude=1.0, epoch=1, stage=0))
    with pytest.raises(ValueError, match="exceeded"):
        codec.encode(np.array([[1.1]]), amplitude=1.0, epoch=1, stage=0)


def test_persistent_gossip_preserves_cache_and_exact_bit_accounting():
    graph = make_static_graph("ring", 4)
    transport = ProgressiveResidualTransport(
        graph, rounds=3, bit_width=8, codec_seed=123, threshold_factor=1.0
    )
    rng = np.random.default_rng(4)
    previous_publications = None
    for epoch in (1, 2):
        values = rng.uniform(-0.5, 0.5, size=(4, 3))
        mixed = transport.run_epoch(values, epoch=epoch)
        assert mixed.shape == values.shape
        assert np.all(np.isfinite(mixed))
        publication, caches = transport.publication, transport.receiver_cache
        for receiver, sender in transport.directed_edges:
            assert np.array_equal(caches[:, receiver, sender], publication[:, sender])
        assert np.max(np.abs(publication)) <= transport.schedule.state_bounds[-1]
        if previous_publications is not None:
            assert not np.array_equal(publication, previous_publications)
        previous_publications = publication
        ledger = transport.ledger
        slots = graph.directed_edge_count * 3 * 3 * epoch
        assert ledger.scheduled_scalar_slots == ledger.flag_bits == slots
        assert ledger.residual_code_bits == ledger.active_residual_events * 8
        assert ledger.payload_bits == ledger.flag_bits + ledger.residual_code_bits
        assert ledger.range_violation_count == 0
