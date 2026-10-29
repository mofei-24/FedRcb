# FedRCB

Reproduction code for FedRCB with additive shared and client-specific linear effects, persistent residual coding, and finite-round peer-to-peer gossip.

## Install

Python 3.11 or newer is required.

## Synthetic data

```bash
fedrcb run --config configs/main.toml --output results/synthetic --workers 8
fedrcb aggregate --output results/synthetic
fedrcb plot --output results/synthetic
```

## MovieLens

```bash
fedrcb prepare movielens --data data/movielens --output data/movielens_bundle.npz
fedrcb run --dataset movielens --bundle data/movielens_bundle.npz --output results/movielens --workers 8
fedrcb aggregate --output results/movielens
fedrcb plot --output results/movielens
```

## Jester

```bash
fedrcb prepare jester --ratings-zip data/jester-data-1.zip --ratings-zip data/jester-data-2.zip --output data/jester_bundle.npz
fedrcb run --dataset jester --bundle data/jester_bundle.npz --output results/jester --workers 8
fedrcb aggregate --output results/jester
fedrcb plot --output results/jester
```