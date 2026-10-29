from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .environment import generate_trial
from .protocol import run_fedrcb


SCHEMA = "fedrcb_trial_v1"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_hash(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def configuration_fingerprint(config: ExperimentConfig) -> str:
    values = asdict(config)
    for name in ("output_dir", "parallel_workers", "bundle_path"):
        values.pop(name, None)
    values["bundle_sha256"] = _file_hash(config.bundle_path) if config.bundle_path else None
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _trial_path(config: ExperimentConfig, heterogeneity: float, seed: int) -> Path:
    level = format(heterogeneity, ".12g").replace(".", "p")
    return Path(config.output_dir) / "trials" / f"h{level}_seed{seed}.json"


def _read_trial(path: Path, fingerprint: str, heterogeneity: float, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("configuration_fingerprint") != fingerprint:
        raise ValueError(f"incompatible result: {path}; use a separate output directory or rerun with --force")
    if payload.get("seed") != seed or payload.get("heterogeneity") != heterogeneity:
        raise ValueError(f"trial identity differs from its filename: {path}")
    return payload


def _generate_trial(config: ExperimentConfig, heterogeneity: float, seed: int):
    if config.dataset == "movielens":
        from .movielens_environment import generate_movielens_trial
        return generate_movielens_trial(config, heterogeneity=heterogeneity, seed=seed)
    if config.dataset == "jester":
        from .jester_environment import generate_jester_trial, load_jester_bundle
        return generate_jester_trial(load_jester_bundle(config.bundle_path), config, seed=seed)
    return generate_trial(config, heterogeneity=heterogeneity, seed=seed)


def _payload(config: ExperimentConfig, trial, trace) -> dict:
    base = trace.base
    rounds = np.asarray(base.checkpoints, dtype=int)
    group = np.asarray(base.group_pseudo_regret, dtype=float)
    target = config.horizon - config.final_window_rounds
    if target:
        indices = np.flatnonzero(rounds == target)
        if indices.size != 1:
            raise RuntimeError("final-window starting checkpoint is missing")
        initial_regret = float(group[indices[0]])
    else:
        initial_regret = 0.0
    total_bits = int(base.communication_bits[-1])
    flag_bits = int(trace.flag_bits[-1])
    code_bits = int(trace.residual_code_bits[-1])
    if total_bits != flag_bits + code_bits:
        raise RuntimeError("communication ledger does not reconcile")
    metrics = {
        "cumulative_regret_per_client_avg": float(group[-1] / config.num_clients),
        "forced_cumulative_regret_per_client_avg": float(base.forced_exploration_group_pseudo_regret[-1] / config.num_clients),
        "policy_cumulative_regret_per_client_avg": float(base.policy_group_pseudo_regret[-1] / config.num_clients),
        "final_window_per_round_per_client_avg": float((group[-1] - initial_regret) / config.final_window_rounds / config.num_clients),
        "communication_bits_total": total_bits,
        "communication_bits_per_client": total_bits / config.num_clients,
        "flag_bits_total": flag_bits,
        "residual_code_bits_total": code_bits,
    }
    return {
        "algorithm": "FedRCB",
        "checkpoints": rounds.tolist(),
        "group_cumulative_regret": group.tolist(),
        "forced_group_cumulative_regret": base.forced_exploration_group_pseudo_regret.tolist(),
        "policy_group_cumulative_regret": base.policy_group_pseudo_regret.tolist(),
        "per_client_cumulative_regret": base.per_client_pseudo_regret.tolist(),
        "communication_bits": base.communication_bits.tolist(),
        "flag_bits": trace.flag_bits.tolist(),
        "residual_code_bits": trace.residual_code_bits.tolist(),
        "scheduled_scalar_slots": trace.scheduled_scalar_slots.tolist(),
        "active_residual_events": trace.active_residual_events.tolist(),
        "range_violation_count": trace.range_violation_count.tolist(),
        "packet_digest": trace.packet_digest,
        "metrics": metrics,
    }


def _worker(task) -> str | None:
    config, heterogeneity, seed, fingerprint, force = task
    path = _trial_path(config, heterogeneity, seed)
    if path.exists() and not force:
        _read_trial(path, fingerprint, heterogeneity, seed)
        return None
    trial = _generate_trial(config, heterogeneity, seed)
    trace = run_fedrcb(trial, config)
    payload = {
        "schema": SCHEMA,
        "configuration_fingerprint": fingerprint,
        "dataset": config.dataset,
        "seed": seed,
        "heterogeneity": heterogeneity,
        "trial_digest": trial.digest,
        **_payload(config, trial, trace),
    }
    _write_json(path, payload)
    return str(path)


def run(config: ExperimentConfig, *, force: bool = False) -> list[Path]:
    config.validate()
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = configuration_fingerprint(config)
    config_path = root / "config.json"
    record_path = root / "run.json"
    if config_path.exists() and not force:
        previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
        if previous.get("configuration_fingerprint") != fingerprint:
            raise ValueError(f"configuration differs from {config_path}; use a separate output directory or --force")
    _write_json(config_path, asdict(config))
    _write_json(record_path, {"schema": "fedrcb_run_v1", "configuration_fingerprint": fingerprint})
    tasks = [(config, float(h), seed, fingerprint, force) for h in config.heterogeneity_values for seed in config.evaluation_seeds]
    written = []
    if config.parallel_workers == 1:
        values = map(_worker, tasks)
        written = [Path(value) for value in values if value is not None]
    else:
        with ProcessPoolExecutor(max_workers=config.parallel_workers) as executor:
            futures = [executor.submit(_worker, task) for task in tasks]
            for future in as_completed(futures):
                value = future.result()
                if value is not None:
                    written.append(Path(value))
    print(f"completed {len(tasks)} trials; wrote {len(written)} result files")
    return sorted(written)


def load_config(output_dir: str | Path) -> ExperimentConfig:
    root = Path(output_dir)
    values = json.loads((root / "config.json").read_text(encoding="utf-8"))
    values["heterogeneity_values"] = tuple(values["heterogeneity_values"])
    values["output_dir"] = str(root)
    config = ExperimentConfig(**values)
    config.validate()
    return config


def _statistics(values: np.ndarray, config: ExperimentConfig) -> dict:
    values = np.asarray(values, dtype=float)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("summary requires finite results from every seed")
    rng = np.random.default_rng(20_260_911)
    draws = rng.choice(values, size=(config.bootstrap_replicas, values.size), replace=True).mean(axis=1)
    alpha = (1 - config.bootstrap_confidence) / 2
    lower, upper = np.quantile(draws, (alpha, 1 - alpha))
    return {"n": int(values.size), "mean": float(values.mean()), "std": float(values.std(ddof=1)) if values.size > 1 else 0.0, "ci_low": float(lower), "ci_high": float(upper)}


def aggregate(config: ExperimentConfig) -> dict:
    config.validate()
    fingerprint = configuration_fingerprint(config)
    root = Path(config.output_dir)
    conditions = []
    metric_rows = []
    trajectory_rows = []
    for heterogeneity in config.heterogeneity_values:
        trials = []
        for seed in config.evaluation_seeds:
            path = _trial_path(config, heterogeneity, seed)
            if not path.is_file():
                raise ValueError(f"missing result: {path}; finish run before aggregating")
            trials.append(_read_trial(path, fingerprint, heterogeneity, seed))
        rounds = trials[0]["checkpoints"]
        if any(value["checkpoints"] != rounds for value in trials):
            raise ValueError("checkpoint grids differ across seeds")
        metrics = {name: _statistics(np.asarray([value["metrics"][name] for value in trials]), config) for name in trials[0]["metrics"]}
        trajectory = {"round": rounds}
        for name in ("group_cumulative_regret", "forced_group_cumulative_regret", "policy_group_cumulative_regret", "communication_bits"):
            values = np.asarray([value[name] for value in trials], dtype=float) / config.num_clients
            trajectory[name + "_mean"] = values.mean(axis=0).tolist()
            trajectory[name + "_std"] = (values.std(axis=0, ddof=1) if len(trials) > 1 else np.zeros(len(rounds))).tolist()
        conditions.append({"heterogeneity": heterogeneity, "seeds": list(config.evaluation_seeds), "metrics": metrics, "trajectory": trajectory})
        for name, result in metrics.items():
            metric_rows.append({"heterogeneity": heterogeneity, "metric": name, **result})
        for index, round_number in enumerate(rounds):
            trajectory_rows.append({"heterogeneity": heterogeneity, "round": round_number, **{name: values[index] for name, values in trajectory.items() if name != "round"}})
    summary = {"schema": "fedrcb_summary_v1", "algorithm": "FedRCB", "dataset": config.dataset, "configuration_fingerprint": fingerprint, "conditions": conditions}
    _write_json(root / "summary.json", summary)
    for filename, rows in (("summary.csv", metric_rows), ("trajectory.csv", trajectory_rows)):
        with (root / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"summary: {root / 'summary.json'}")
    return summary
