from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest

from fedrcb.config import ExperimentConfig
from fedrcb.runner import aggregate, configuration_fingerprint, run


def _config(tmp_path: Path, **updates) -> ExperimentConfig:
    values = dict(num_clients=4, num_actions=4, global_dim=2, local_dim=2, horizon=20, epoch_length=10, checkpoint_interval=10, library_size=12, gossip_rounds=2, bit_width=8, heterogeneity_values=(0.1,), evaluation_seed_count=1, final_window_rounds=10, bootstrap_replicas=20, output_dir=str(tmp_path))
    values.update(updates)
    return ExperimentConfig(**values)


def test_resume_requires_the_original_algorithm_parameters(tmp_path):
    config = _config(tmp_path)
    assert len(run(config)) == 1
    assert run(config) == []
    with pytest.raises(ValueError, match="configuration differs"):
        run(replace(config, bit_width=12))


def test_output_relocation_and_worker_count_preserve_resume(tmp_path):
    config = _config(tmp_path / "original")
    run(config)
    destination = tmp_path / "moved"
    shutil.copytree(config.output_dir, destination)
    relocated = replace(config, output_dir=str(destination))
    assert configuration_fingerprint(config) == configuration_fingerprint(replace(relocated, parallel_workers=2))
    assert run(relocated) == []


def test_aggregation_requires_every_configured_seed(tmp_path):
    config = _config(tmp_path, evaluation_seed_count=2)
    paths = run(config)
    paths[-1].unlink()
    with pytest.raises(ValueError, match="missing result"):
        aggregate(config)


def test_total_regret_and_payload_preserve_their_decompositions(tmp_path):
    config = _config(tmp_path)
    run(config)
    summary = aggregate(config)
    metrics = summary["conditions"][0]["metrics"]
    assert np.isclose(metrics["cumulative_regret_per_client_avg"]["mean"], metrics["forced_cumulative_regret_per_client_avg"]["mean"] + metrics["policy_cumulative_regret_per_client_avg"]["mean"])
    assert metrics["communication_bits_total"]["mean"] == metrics["flag_bits_total"]["mean"] + metrics["residual_code_bits_total"]["mean"]


def test_a_full_horizon_window_starts_from_zero_regret(tmp_path):
    config = _config(tmp_path, final_window_rounds=20)
    run(config)
    metrics = aggregate(config)["conditions"][0]["metrics"]
    assert np.isclose(metrics["final_window_per_round_per_client_avg"]["mean"] * config.horizon, metrics["cumulative_regret_per_client_avg"]["mean"])
