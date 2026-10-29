from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .context import (
    ClientContextDiagnostics,
    ContextDiagnostics,
    PublicClientContextProfile,
    PublicContextLibrary,
    context_diagnostics,
    context_diagnostics_by_client,
)
from .environment import LinearBanditTrial, _trial_digest


@dataclass(frozen=True)
class MovieLensBundle:


    x_global: np.ndarray
    x_local: np.ndarray
    client_probabilities: np.ndarray
    theta_global: np.ndarray
    delta_natural: np.ndarray
    noise_std: float
    h_natural: float
    movie_ids: np.ndarray
    manifest: dict
    digest: str

    @property
    def library_size(self) -> int:
        return int(self.x_global.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.x_global.shape[1])

    @property
    def num_clients(self) -> int:
        return int(self.client_probabilities.shape[0])


def _load_tables(data_dir: Path, *, min_movie_ratings: int):
    import pandas as pd

    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")


    if ratings["timestamp"].dtype == object:
        ratings["timestamp"] = (
            ratings["timestamp"].astype(str).str.strip().astype("int64")
        )
    counts = ratings.groupby("movieId").size()
    kept = counts[counts >= min_movie_ratings].index
    ratings = ratings[ratings["movieId"].isin(kept)].reset_index(drop=True)
    movies = movies[movies["movieId"].isin(kept)].reset_index(drop=True)
    return ratings, movies


_YEAR = re.compile(r"\((\d{4})\)\s*$")


def _movie_content_matrix(movies):


    genres = sorted(
        {g for raw in movies["genres"] for g in str(raw).split("|") if g}
    )
    genre_index = {g: i for i, g in enumerate(genres)}
    years: list[int] = []
    for title in movies["title"]:
        match = _YEAR.search(str(title))
        years.append(int(match.group(1)) if match else 1990)
    decades = sorted({y // 10 * 10 for y in years})
    decade_index = {d: i for i, d in enumerate(decades)}
    rows = np.zeros((len(movies), len(genres) + len(decades)), dtype=float)
    movie_ids = movies["movieId"].to_numpy()
    id_to_row = {int(mid): i for i, mid in enumerate(movie_ids)}
    for i, raw in enumerate(movies["genres"]):
        for g in str(raw).split("|"):
            if g in genre_index:
                rows[i, genre_index[g]] = 1.0
    for i, y in enumerate(years):
        rows[i, len(genres) + decade_index[y // 10 * 10]] = 1.0
    return rows, id_to_row, genres


def build_movielens_bundle(
    data_dir: str | Path,
    *,
    num_clients: int = 10,
    library_size: int = 100,
    num_actions: int = 10,
    global_dim: int = 5,
    local_dim: int = 5,
    svd_components: int = 16,
    context_norm: float = 1.0,
    local_energy_fraction: float = 0.75,
    global_theta_norm: float = 0.8,
    min_context_eigenvalue: float = 0.02,
    min_eigenvalue_floor: float = 0.005,
    min_movie_ratings: int = 50,
    cross_genre_fraction: float = 0.3,
    slate_smoothing: float = 1.0,
    ridge_alpha: float = 1.0,
    test_fraction: float = 0.2,
    seed: int = 202_609,
    max_slate_attempts: int = 200,
) -> MovieLensBundle:


    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA, TruncatedSVD

    rng = np.random.default_rng(seed)
    data_path = Path(data_dir)
    ratings, movies = _load_tables(data_path, min_movie_ratings=min_movie_ratings)
    content, id_to_row, genre_names = _movie_content_matrix(movies)
    movie_ids_all = movies["movieId"].to_numpy()


    ratings = ratings.sort_values(["userId", "timestamp"], kind="mergesort")
    group_rank = ratings.groupby("userId").cumcount()
    group_size = ratings["userId"].map(ratings.groupby("userId").size())
    is_test = (group_rank >= ((1.0 - test_fraction) * group_size).astype(int)).to_numpy()
    ratings_train = ratings[~is_test]
    user_ids = np.sort(ratings["userId"].unique())
    user_index = {int(u): i for i, u in enumerate(user_ids)}
    n_users = len(user_ids)


    row_of_movie = np.full(int(movie_ids_all.max()) + 1, -1, dtype=np.int64)
    for mid, row in id_to_row.items():
        row_of_movie[int(mid)] = row
    n_movies = len(movie_ids_all)
    matrix = np.zeros((n_users, n_movies), dtype=np.float32)
    user_mean = np.zeros(n_users, dtype=np.float32)
    train_u = ratings_train["userId"].map(user_index).to_numpy()
    train_m = row_of_movie[ratings_train["movieId"].to_numpy()]
    train_r = ((ratings_train["rating"].to_numpy() - 3.0) / 2.0).astype(np.float32)
    counts = np.bincount(train_u, minlength=n_users).astype(np.float32)
    sums = np.bincount(train_u, weights=train_r, minlength=n_users).astype(np.float32)
    user_mean = sums / np.maximum(counts, 1.0)
    matrix[train_u, train_m] = train_r - user_mean[train_u]
    svd = TruncatedSVD(n_components=svd_components, random_state=seed)
    user_factors = svd.fit_transform(matrix)
    movie_factors = svd.components_.T


    kmeans = KMeans(n_clusters=num_clients, n_init=10, random_state=seed)
    cluster_of_user = kmeans.fit_predict(user_factors)


    pca_g = PCA(n_components=global_dim, random_state=seed).fit(content)
    v_g = pca_g.transform(content)
    v_l = movie_factors[:, :local_dim].astype(float)

    norm_l = np.linalg.norm(v_l, axis=1)
    usable = norm_l > 1e-8
    v_g = v_g[usable]
    v_l = v_l[usable]
    movie_ids_kept = movie_ids_all[usable]
    id_to_arm = {int(mid): i for i, mid in enumerate(movie_ids_kept)}
    g_hat = v_g / np.maximum(np.linalg.norm(v_g, axis=1, keepdims=True), 1e-12)
    l_hat = v_l / np.maximum(norm_l[usable, None], 1e-12)
    x_g_arm = context_norm * np.sqrt(1.0 - local_energy_fraction) * g_hat
    x_l_arm = context_norm * np.sqrt(local_energy_fraction) * l_hat


    target_train = train_r.astype(float) - user_mean[train_u].astype(float)
    arm_of_train = np.array([id_to_arm[int(m)] for m in ratings_train["movieId"]])
    feats_g = x_g_arm[arm_of_train]
    gram_g = feats_g.T @ feats_g + ridge_alpha * np.eye(global_dim)
    theta_raw = np.linalg.solve(gram_g, feats_g.T @ target_train)


    theta_norm_raw = float(np.linalg.norm(theta_raw))
    if theta_norm_raw <= global_theta_norm:
        theta_global = theta_raw
    else:
        theta_global = theta_raw * (global_theta_norm / max(theta_norm_raw, 1e-12))
    resid = target_train - feats_g @ theta_global
    delta_raw = np.zeros((num_clients, local_dim))
    cluster_of_train = cluster_of_user[train_u]
    feats_l = x_l_arm[arm_of_train]
    for client in range(num_clients):
        mask = cluster_of_train == client
        if mask.sum() < local_dim + 1:
            continue
        block = feats_l[mask]
        gram = block.T @ block + ridge_alpha * np.eye(local_dim)
        delta_raw[client] = np.linalg.solve(gram, block.T @ resid[mask])


    delta_mean = delta_raw.mean(axis=0, keepdims=True)
    h_natural = float(np.sqrt(np.mean(np.sum((delta_raw - delta_mean) ** 2, axis=1))))
    delta_common_norm = float(np.linalg.norm(delta_mean))


    genre_sets = [
        {
            id_to_arm[int(mid)]
            for mid, raw in zip(movie_ids_kept, movies["genres"][usable])
            if genre in str(raw).split("|")
        }
        for genre in genre_names
    ]
    pools = [s for s in genre_sets if len(s) >= num_actions]
    pool_names = [g for g, s in zip(genre_names, genre_sets) if len(s) >= num_actions]
    if len(pools) < 2:
        raise RuntimeError("fewer than two genre pools can fill a slate")
    popularity = np.bincount(arm_of_train, minlength=len(movie_ids_kept)).astype(float)


    client_of_arm_mass = np.zeros((num_clients, len(movie_ids_kept)))
    np.add.at(
        client_of_arm_mass,
        (cluster_of_train, arm_of_train),
        1.0,
    )

    def client_laws(slates: np.ndarray) -> np.ndarray:
        mass = client_of_arm_mass[:, slates].sum(axis=2)
        probs = (mass + slate_smoothing) / (
            mass.sum(axis=1, keepdims=True) + slate_smoothing * library_size
        )
        return probs


    anchor_counts = np.array(
        [max(1, round(library_size * len(p) / sum(len(q) for q in pools))) for p in pools]
    )
    while anchor_counts.sum() < library_size:
        anchor_counts[np.argmax([len(p) for p in pools])] += 1
    while anchor_counts.sum() > library_size:
        anchor_counts[np.argmax(anchor_counts)] -= 1
    anchors = np.repeat(np.arange(len(pools)), anchor_counts)

    best: tuple[float, np.ndarray, np.ndarray, ClientContextDiagnostics] | None = None
    diagnostics = None
    slates = None
    for attempt in range(max_slate_attempts):
        slates = np.empty((library_size, num_actions), dtype=np.int64)
        all_arms = np.arange(len(movie_ids_kept))
        for ell, pool_idx in enumerate(anchors):
            pool = np.array(sorted(pools[int(pool_idx)]))
            n_anchor = max(1, int(round(num_actions * (1.0 - cross_genre_fraction))))
            weights_anchor = popularity[pool] + 1e-6
            chosen = rng.choice(
                pool, size=n_anchor, replace=False, p=weights_anchor / weights_anchor.sum()
            )
            rest = np.setdiff1d(all_arms, chosen)
            weights_rest = popularity[rest] + 1e-6
            fill = rng.choice(
                rest,
                size=num_actions - n_anchor,
                replace=False,
                p=weights_rest / weights_rest.sum(),
            )
            slates[ell] = np.concatenate([chosen, fill])
        x_global = x_g_arm[slates]
        x_local = x_l_arm[slates]
        probs = client_laws(slates)
        pooled = probs.mean(axis=0)
        diagnostics = context_diagnostics_by_client(
            x_global, x_local, sampling_probabilities=probs, base_probabilities=pooled
        )
        worst = min(
            diagnostics.global_min_eigenvalue,
            diagnostics.local_min_eigenvalue,
            diagnostics.full_min_eigenvalue,
        )
        if best is None or worst > best[0]:
            best = (worst, slates.copy(), probs.copy(), diagnostics)
        if worst >= min_context_eigenvalue:
            break
    assert best is not None and diagnostics is not None and slates is not None
    achieved = min(
        diagnostics.global_min_eigenvalue,
        diagnostics.local_min_eigenvalue,
        diagnostics.full_min_eigenvalue,
    )
    if achieved < min_context_eigenvalue:
        achieved, slates, probs, diagnostics = best
        if achieved < min_eigenvalue_floor:
            raise RuntimeError(
                f"slate design cannot reach the eigenvalue floor "
                f"({achieved:.4g} < {min_eigenvalue_floor:.4g})"
            )
        x_global = x_g_arm[slates]
        x_local = x_l_arm[slates]
    pooled = probs.mean(axis=0)
    pooled_diag = context_diagnostics(
        x_global, x_local, sampling_probabilities=pooled
    )


    ratings_test = ratings[is_test]
    arm_of_test = np.array(
        [id_to_arm.get(int(m), -1) for m in ratings_test["movieId"]], dtype=int
    )
    valid = arm_of_test >= 0
    arm_of_test = arm_of_test[valid]
    test_u = ratings_test["userId"].map(user_index).to_numpy()[valid]

    test_r = (
        (ratings_test["rating"].to_numpy() - 3.0) / 2.0
    )[valid] - user_mean[test_u].astype(float)

    pred = (
        x_g_arm[arm_of_test] @ theta_global
        + np.einsum(
            "ij,ij->i",
            x_l_arm[arm_of_test],
            delta_raw[cluster_of_user[test_u]],
        )
    )
    residual = test_r - pred
    noise_std = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((test_r - test_r.mean()) ** 2))
    mae = float(np.mean(np.abs(residual)))

    manifest = {
        "source": "User-supplied MovieLens-format ratings.csv and movies.csv",
        "ratings_after_filter": int(len(ratings)),
        "rated_movies_after_filter": int(ratings["movieId"].nunique()),
        "pipeline_seed": int(seed),
        "num_clients": int(num_clients),
        "library_size": int(library_size),
        "num_actions": int(num_actions),
        "global_dim": int(global_dim),
        "local_dim": int(local_dim),
        "svd_components": int(svd_components),
        "context_norm": float(context_norm),
        "local_energy_fraction": float(local_energy_fraction),
        "global_theta_norm": float(global_theta_norm),
        "theta_norm_achieved": float(np.linalg.norm(theta_global)),
        "theta_norm_capped": bool(theta_norm_raw > global_theta_norm),
        "min_context_eigenvalue_requested": float(min_context_eigenvalue),
        "min_context_eigenvalue_achieved": float(achieved),
        "eigenvalue_relaxed": bool(achieved < min_context_eigenvalue),
        "min_movie_ratings": int(min_movie_ratings),
        "cross_genre_fraction": float(cross_genre_fraction),
        "slate_smoothing": float(slate_smoothing),
        "ridge_alpha": float(ridge_alpha),
        "test_fraction": float(test_fraction),
        "rating_transform": "(rating - 3) / 2 then per-user train-mean removal",
        "h_natural": float(h_natural),
        "delta_common_norm": delta_common_norm,
        "heterogeneity_semantics": (
            "h rescales only the cluster deviations (delta_i - delta_mean); "
            "the common collaborative component delta_mean is always kept. "
            "h=0 makes all clients identical (shared delta_mean); "
            "h=h_natural reproduces the as-fitted surrogate."
        ),
        "noise_std": float(noise_std),
        "surrogate_test_mae": mae,
        "surrogate_test_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "surrogate_test_n": int(valid.sum()),
        "movies_kept": int(len(movie_ids_kept)),
        "users_total": int(n_users),
        "genre_pool_names": pool_names,
        "surrogate_note": (
            "Targets are per-user train-mean-removed ratings: the user rating "
            "bias is a per-user constant shift across arms (regret-invariant "
            "and unlearnable without user identity), so it is excluded from "
            "the surrogate rather than counted as method-facing noise.  "
            "Rewards are surrogate predictions of the deployed (norm-rescaled) "
            "ridge model plus N(0, noise_std^2); pseudo-regret is measured "
            "against the surrogate. Client slate laws are train-split cluster "
            "rating masses, epsilon-smoothed."
        ),
    }
    hasher = hashlib.sha256()
    hasher.update(
        (
            f"kind=movielens;L={library_size};K={num_actions};dg={global_dim};"
            f"dl={local_dim};n={num_clients};seed={seed}".encode("ascii")
        )
    )
    for array in (x_global, x_local, probs, theta_global, delta_raw):
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    digest = hasher.hexdigest()
    for array in (x_global, x_local, probs, theta_global, delta_raw):
        array.setflags(write=False)
    return MovieLensBundle(
        x_global=x_global,
        x_local=x_local,
        client_probabilities=probs,
        theta_global=theta_global,
        delta_natural=delta_raw,
        noise_std=noise_std,
        h_natural=h_natural,
        movie_ids=movie_ids_kept[slates],
        manifest=manifest,
        digest=digest,
    )


def save_bundle(bundle: MovieLensBundle, path: str | Path) -> Path:


    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        x_global=bundle.x_global,
        x_local=bundle.x_local,
        client_probabilities=bundle.client_probabilities,
        theta_global=bundle.theta_global,
        delta_natural=bundle.delta_natural,
        movie_ids=bundle.movie_ids,
    )
    np.savez_compressed(path, **payload)
    manifest = dict(bundle.manifest)
    manifest["bundle_digest"] = bundle.digest
    manifest["npz_name"] = path.name
    with path.with_suffix(".manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return path


_BUNDLE_CACHE: dict[str, MovieLensBundle] = {}


def load_bundle(path: str | Path) -> MovieLensBundle:


    key = str(Path(path).resolve())
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    path = Path(key)
    if not path.exists():
        raise FileNotFoundError(f"frozen MovieLens bundle not found: {path}")
    with np.load(path) as data:
        arrays = {name: np.array(data[name]) for name in data.files}
    with path.with_suffix(".manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    x_global = arrays["x_global"]
    x_local = arrays["x_local"]
    probs = arrays["client_probabilities"]
    theta = arrays["theta_global"]
    delta = arrays["delta_natural"]
    hasher = hashlib.sha256()
    hasher.update(
        (
            "kind=movielens;"
            + f"L={x_global.shape[0]};K={x_global.shape[1]};"
            + f"dg={x_global.shape[2]};dl={x_local.shape[2]};"
            + f"n={probs.shape[0]};seed={manifest['pipeline_seed']}"
        ).encode("ascii")
    )
    for array in (x_global, x_local, probs, theta, delta):
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    digest = hasher.hexdigest()
    if digest != manifest["bundle_digest"]:
        raise RuntimeError(
            "frozen MovieLens bundle digest mismatch "
            f"({digest} != {manifest['bundle_digest']})"
        )
    bundle = MovieLensBundle(
        x_global=x_global,
        x_local=x_local,
        client_probabilities=probs,
        theta_global=theta,
        delta_natural=delta,
        noise_std=float(manifest["noise_std"]),
        h_natural=float(manifest["h_natural"]),
        movie_ids=arrays["movie_ids"],
        manifest=manifest,
        digest=digest,
    )
    for array in (
        bundle.x_global,
        bundle.x_local,
        bundle.client_probabilities,
        bundle.theta_global,
        bundle.delta_natural,
    ):
        array.setflags(write=False)
    _BUNDLE_CACHE[key] = bundle
    return bundle


def _profile_from_bundle(bundle: MovieLensBundle) -> PublicClientContextProfile:
    pooled = bundle.client_probabilities.mean(axis=0)
    hasher = hashlib.sha256()
    hasher.update(
        (
            "kind=movielens-profile;"
            + f"n={bundle.num_clients};L={bundle.library_size}"
        ).encode("ascii")
    )
    for array in (pooled, bundle.client_probabilities):
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    profile = PublicClientContextProfile(
        base_probabilities=pooled,
        preference_probabilities=bundle.client_probabilities,
        sampling_probabilities=bundle.client_probabilities,
        mixture_eta=1.0,
        profile_strength=1.0,
        profile_seed=int(bundle.manifest["pipeline_seed"]),
        digest=hasher.hexdigest(),
        kind="movielens",
        concentration=None,
    )
    for array in (pooled, bundle.client_probabilities):
        array.setflags(write=False)
    return profile


def _library_from_bundle(bundle: MovieLensBundle) -> PublicContextLibrary:
    pooled = bundle.client_probabilities.mean(axis=0)
    diagnostics = context_diagnostics(
        bundle.x_global, bundle.x_local, sampling_probabilities=pooled
    )
    hasher = hashlib.sha256()
    hasher.update(
        (
            "kind=movielens-library;"
            + f"L={bundle.library_size};K={bundle.num_actions}"
        ).encode("ascii")
    )
    for array in (bundle.x_global, bundle.x_local, pooled):
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    library = PublicContextLibrary(
        x_global=bundle.x_global,
        x_local=bundle.x_local,
        context_norm=float(bundle.manifest["context_norm"]),
        sampling_law="movielens_real",
        sampling_probabilities=pooled,
        diagnostics=diagnostics,
        digest=hasher.hexdigest(),
        local_energy_fraction=float(bundle.manifest["local_energy_fraction"]),
    )
    pooled.setflags(write=False)
    return library


def generate_movielens_trial(
    config,
    *,
    heterogeneity: float,
    seed: int,
) -> LinearBanditTrial:


    if getattr(config, "context_mode", None) != "movielens":
        raise ValueError("generate_movielens_trial requires context_mode='movielens'")
    if heterogeneity < 0:
        raise ValueError("heterogeneity must be non-negative")
    bundle = load_bundle(config.movielens_env_path)
    n = bundle.num_clients
    horizon = int(config.horizon)
    library_size = bundle.library_size
    if config.num_actions != bundle.num_actions:
        raise ValueError("config.num_actions must match the frozen bundle")
    if config.global_dim != bundle.x_global.shape[2]:
        raise ValueError("config.global_dim must match the frozen bundle")
    if config.local_dim != bundle.x_local.shape[2]:
        raise ValueError("config.local_dim must match the frozen bundle")
    if config.num_clients != n:
        raise ValueError("config.num_clients must match the frozen bundle")

    seed_sequence = np.random.SeedSequence(seed)
    _, _, tape_seed = seed_sequence.spawn(3)
    tape_rng = np.random.default_rng(tape_seed)

    profile = _profile_from_bundle(bundle)
    library = _library_from_bundle(bundle)
    client_context_diagnostics = context_diagnostics_by_client(
        library.x_global,
        library.x_local,
        sampling_probabilities=profile.sampling_probabilities,
        base_probabilities=library.sampling_probabilities,
    )

    if heterogeneity == 0.0:


        delta_local = np.repeat(
            bundle.delta_natural.mean(axis=0, keepdims=True), n, axis=0
        )
    else:

        delta_mean = bundle.delta_natural.mean(axis=0, keepdims=True)
        scale = float(heterogeneity) / bundle.h_natural
        delta_local = delta_mean + (bundle.delta_natural - delta_mean) * scale
    theta_global = bundle.theta_global.copy()

    client_context_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 908_117])
    )
    client_context_uniforms = client_context_rng.random(size=(n, horizon))
    cdf = np.cumsum(profile.sampling_probabilities, axis=1)
    context_ids = np.empty((n, horizon), dtype=np.int64)
    for client in range(n):
        context_ids[client] = np.searchsorted(
            cdf[client], client_context_uniforms[client], side="right"
        )
    if np.any(context_ids < 0) or np.any(context_ids >= library_size):
        raise RuntimeError("inverse-CDF context sampler emitted an invalid row index")
    reward_noise = tape_rng.normal(
        loc=0.0, scale=float(config.noise_std), size=(n, horizon, bundle.num_actions)
    )
    exploration_uniforms = tape_rng.random(size=(n, horizon))
    exploration_actions = tape_rng.integers(
        0, bundle.num_actions, size=(n, horizon), endpoint=False
    )
    for array in (
        theta_global,
        delta_local,
        context_ids,
        reward_noise,
        exploration_uniforms,
        exploration_actions,
        client_context_uniforms,
    ):
        array.setflags(write=False)
    digest = _trial_digest(
        library.digest,
        theta_global,
        delta_local,
        context_ids,
        reward_noise,
        exploration_uniforms,
        exploration_actions,
        context_profile_digest=profile.digest,
        client_context_uniforms=client_context_uniforms,
    )
    return LinearBanditTrial(
        library=library,
        client_context_profile=profile,
        client_context_diagnostics=client_context_diagnostics,
        theta_global=theta_global,
        delta_local=delta_local,
        context_ids=context_ids,
        reward_noise=reward_noise,
        exploration_uniforms=exploration_uniforms,
        exploration_actions=exploration_actions,
        client_context_uniforms=client_context_uniforms,
        seed=int(seed),
        heterogeneity=float(heterogeneity),
        digest=digest,
    )
