from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import tomllib

from .config import ExperimentConfig
from .runner import aggregate, load_config, run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedrcb", description="Reproduce the FedRCB method")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "smoke"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", choices=("synthetic", "movielens", "jester"))
        sub.add_argument("--config", type=Path)
        sub.add_argument("--bundle", type=Path)
        sub.add_argument("--output", type=Path, default=Path("results") / ("smoke" if command == "smoke" else "main"))
        sub.add_argument("--workers", type=int, default=1)
        sub.add_argument("--horizon", type=int)
        sub.add_argument("--seeds", type=int)
        sub.add_argument("--clients", type=int)
        sub.add_argument("--h-values", type=float, nargs="+")
        sub.add_argument("--force", action="store_true")
    for command in ("aggregate", "plot"):
        sub = commands.add_parser(command)
        sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("prepare")
    sub.add_argument("dataset", choices=("movielens", "jester"))
    sub.add_argument("--data", type=Path)
    sub.add_argument("--ratings-zip", type=Path, action="append")
    sub.add_argument("--output", type=Path, required=True)
    return parser


def _configuration(args: argparse.Namespace) -> ExperimentConfig:
    values = {}
    record = args.output / "config.json"
    if record.is_file():
        values.update(json.loads(record.read_text(encoding="utf-8")))
    if args.config:
        with args.config.open("rb") as stream:
            parsed = tomllib.load(stream)
        values.update(parsed.get("experiment", parsed))
    if args.dataset:
        values["dataset"] = args.dataset
    dataset = values.get("dataset", "synthetic")
    if args.bundle:
        values["bundle_path"] = str(args.bundle.resolve())
    if dataset == "jester":
        values.setdefault("heterogeneity_values", (0.3,))
        values.setdefault("evaluation_seed_base", 202_610_301)
    if args.command == "smoke":
        values.update(horizon=1000 if dataset == "jester" else 200, evaluation_seed_count=1, bootstrap_replicas=100)
    for argument, field in (("horizon", "horizon"), ("seeds", "evaluation_seed_count"), ("clients", "num_clients")):
        value = getattr(args, argument)
        if value is not None:
            values[field] = value
    if args.h_values is not None:
        values["heterogeneity_values"] = tuple(args.h_values)
    if dataset != "synthetic":
        if not values.get("bundle_path"):
            raise ValueError(f"{dataset} requires --bundle")
        values["bundle_path"] = str(Path(values["bundle_path"]).resolve())
    if dataset == "movielens":
        from .movielens_environment import load_bundle
        bundle = load_bundle(values["bundle_path"])
        values.update(num_clients=bundle.num_clients, num_actions=bundle.num_actions, global_dim=bundle.x_global.shape[2], local_dim=bundle.x_local.shape[2], library_size=bundle.library_size, noise_std=float(bundle.noise_std), ucb_noise_proxy=float(bundle.noise_std), min_context_eigenvalue=float(bundle.manifest["min_context_eigenvalue_achieved"]))
        if args.h_values is None and "heterogeneity_values" not in values:
            values["heterogeneity_values"] = tuple(sorted({0.0, 0.1, 0.3, round(float(bundle.h_natural), 6)}))
    elif dataset == "jester":
        from .jester_environment import load_jester_bundle
        bundle = load_jester_bundle(values["bundle_path"])
        values.update(num_clients=10, num_actions=10, global_dim=5, local_dim=5, library_size=100, noise_std=float(bundle.train_noise_proxy), ucb_noise_proxy=float(bundle.train_noise_proxy), reward_clip=1.0)
        if tuple(values["heterogeneity_values"]) != (0.3,):
            raise ValueError("Jester uses its observed fixed client heterogeneity; --h-values must be 0.3")
    horizon = values.get("horizon", 20_000)
    interval = values.get("checkpoint_interval", 100)
    if "final_window_rounds" not in values or values["final_window_rounds"] > horizon or args.command == "smoke":
        values["final_window_rounds"] = min(horizon, max(interval, (horizon // 10 // interval) * interval))
    values.update(output_dir=str(args.output), parallel_workers=args.workers)
    valid = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError("unknown configuration fields: " + ", ".join(unknown))
    if "heterogeneity_values" in values:
        values["heterogeneity_values"] = tuple(values["heterogeneity_values"])
    config = ExperimentConfig(**values)
    config.validate()
    return config


def _prepare(args: argparse.Namespace) -> None:
    if args.dataset == "movielens":
        if args.data is None:
            raise ValueError("MovieLens preparation requires --data with ratings.csv and movies.csv")
        from .movielens_environment import build_movielens_bundle, save_bundle
        bundle = build_movielens_bundle(args.data, num_clients=10, library_size=100, num_actions=10, global_dim=5, local_dim=5, svd_components=16, context_norm=1.0, local_energy_fraction=0.75, global_theta_norm=0.8, min_context_eigenvalue=0.02, min_eigenvalue_floor=0.005, min_movie_ratings=50, cross_genre_fraction=0.3, slate_smoothing=1.0, ridge_alpha=1.0, test_fraction=0.2, seed=202_609)
        save_bundle(bundle, args.output)
    else:
        from .jester_environment import build_jester_bundle, save_jester_bundle
        if not args.ratings_zip:
            raise ValueError("Jester preparation requires both official Dataset 1 rating archives")
        bundle = build_jester_bundle(args.ratings_zip)
        save_jester_bundle(bundle, args.output)
    print(f"bundle: {args.output}; digest: {bundle.digest}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        _prepare(args)
    elif args.command in ("run", "smoke"):
        config = _configuration(args)
        run(config, force=args.force)
        if args.command == "smoke":
            aggregate(config)
    elif args.command == "aggregate":
        aggregate(load_config(args.output))
    else:
        from .plotting import plot
        for path in plot(args.output):
            print(f"figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
