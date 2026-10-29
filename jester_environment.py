from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Sequence
import uuid
import zipfile

import numpy as np

from .context import (
    PublicClientContextProfile,
    PublicContextLibrary,
    context_diagnostics,
    context_diagnostics_by_client,
)
from .environment import LinearBanditTrial


GAUGE_JOKE_IDS = (5, 7, 8, 13, 15, 16, 17, 18, 19, 20)
PIPELINE_SEED = 20261003
SOURCE_URL = "https://eigentaste.berkeley.edu/dataset/"
SCHEMA = "jester-real-ratings-v1"
_ARRAY_NAMES = (
    "x_global", "x_local", "slate_item_ids", "item_joke_ids",
    "client_probabilities", "train_ratings", "train_gauge", "train_groups",
    "train_user_ids", "heldout_ratings", "heldout_gauge", "heldout_groups",
    "heldout_user_ids", "train_group_means", "heldout_means", "gauge_centers",
)


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _array_hash(hasher, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(json.dumps(array.shape).encode("ascii"))
    hasher.update(array.view(np.uint8))


def _digest(arrays: dict[str, np.ndarray], manifest: dict) -> str:
    hasher = hashlib.sha256()
    clean = {key: value for key, value in manifest.items() if key != "bundle_digest"}
    hasher.update(json.dumps(clean, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8"))
    for name in _ARRAY_NAMES:
        _array_hash(hasher, name, arrays[name])
    return hasher.hexdigest()


@dataclass(frozen=True)
class JesterBundle:


    x_global: np.ndarray
    x_local: np.ndarray
    slate_item_ids: np.ndarray
    item_joke_ids: np.ndarray
    client_probabilities: np.ndarray
    train_ratings: np.ndarray
    train_gauge: np.ndarray
    train_groups: np.ndarray
    train_user_ids: np.ndarray
    heldout_ratings: np.ndarray
    heldout_gauge: np.ndarray
    heldout_groups: np.ndarray
    heldout_user_ids: np.ndarray
    train_group_means: np.ndarray
    heldout_means: np.ndarray
    gauge_centers: np.ndarray
    manifest: dict
    digest: str

    @property
    def train_noise_proxy(self) -> float:
        return float(self.manifest["train_noise_proxy"])

    @property
    def num_clients(self) -> int:
        return int(self.client_probabilities.shape[0])

    @property
    def library_size(self) -> int:
        return int(self.slate_item_ids.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.slate_item_ids.shape[1])


def _read_rating_zip(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:

    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError("Jester preparation requires xlrd>=2 for official .xls files") from exc
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist()
                   if name.lower().endswith(".xls") and not name.startswith("__MACOSX/")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected exactly one official Dataset 1 XLS")
        member = members[0]
        workbook = xlrd.open_workbook(file_contents=archive.read(member), on_demand=True)
        try:
            sheet = workbook.sheet_by_index(0)
            if workbook.nsheets != 1 or sheet.ncols != 101:
                raise ValueError(f"{path}: expected one 101-column official sheet")


            part_by_rows = {24983: "1_1", 23500: "1_2"}
            if sheet.nrows not in part_by_rows:
                raise ValueError(f"{path}: {sheet.nrows} rows is not Dataset 1_1 or 1_2")
            part = part_by_rows[sheet.nrows]
            raw = np.asarray([sheet.row_values(row) for row in range(sheet.nrows)], dtype=float)
        finally:
            workbook.release_resources()
    ratings = raw[:, 1:]
    rated = ratings != 99.0
    if not np.all(np.isfinite(raw)) or np.any(rated & ((ratings < -10.0) | (ratings > 10.0))):
        raise ValueError(f"{path}: non-finite or invalid rating values")
    if not np.array_equal(raw[:, 0], np.sum(rated, axis=1)):
        raise ValueError(f"{path}: reported and observed rating counts disagree")
    complete = np.all(rated, axis=1)
    source_rows = np.flatnonzero(complete) + 1
    user_ids = np.asarray([f"{part}:{row}" for row in source_rows], dtype="U24")
    selected = np.asarray(ratings[complete] / 10.0, dtype=np.float64)
    return selected, user_ids, {
        "part": part, "filename": path.name, "sha256": file_sha256(path),
        "xls_member": member, "rows_original": int(raw.shape[0]),
        "rows_complete": int(np.sum(complete)),
        "rows_excluded_incomplete": int(np.sum(~complete)),
    }


def _normalize_block(raw: np.ndarray, squared_energy: float) -> np.ndarray:
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("A feature block has a degenerate item; no outcome-based retry is allowed")
    return np.sqrt(squared_energy) * raw / norms


def build_jester_bundle(rating_zips: Sequence[str | Path], *,
                        seed: int = PIPELINE_SEED, train_fraction: float = 0.6,
                        slate_smoothing: float = 0.05) -> JesterBundle:

    from sklearn import __version__ as sklearn_version
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD

    if seed != PIPELINE_SEED or train_fraction != 0.6 or slate_smoothing != 0.05:
        raise ValueError("Jester preparation uses seed=20261003, train_fraction=.6, smoothing=.05")
    if len(rating_zips) != 2:
        raise ValueError("Supply exactly Dataset 1_1 and Dataset 1_2 archives")
    loaded = [_read_rating_zip(Path(path)) for path in rating_zips]
    loaded.sort(key=lambda value: value[2]["part"])
    if [value[2]["part"] for value in loaded] != ["1_1", "1_2"]:
        raise ValueError("The two source archives must contain different official parts 1_1 and 1_2")
    full = np.concatenate([value[0] for value in loaded])
    complete_ids = np.concatenate([value[1] for value in loaded])
    if full.shape[0] < 30:
        raise ValueError("Too few complete cases for a fixed ten-client train/heldout split")
    permutation = np.random.default_rng(seed).permutation(full.shape[0])
    cut = int(np.floor(train_fraction * full.shape[0]))
    train_idx, heldout_idx = permutation[:cut], permutation[cut:]
    gauge_idx = np.asarray(GAUGE_JOKE_IDS, dtype=np.int64) - 1
    item_ids = np.asarray([item for item in range(1, 101) if item not in GAUGE_JOKE_IDS], dtype=np.int64)
    item_idx = item_ids - 1
    train_gauge = full[train_idx][:, gauge_idx]
    heldout_gauge = full[heldout_idx][:, gauge_idx]
    kmeans = KMeans(n_clusters=10, n_init=10, random_state=seed)
    train_groups = kmeans.fit_predict(train_gauge).astype(np.int64)
    heldout_groups = kmeans.predict(heldout_gauge).astype(np.int64)
    train_counts = np.bincount(train_groups, minlength=10)
    heldout_counts = np.bincount(heldout_groups, minlength=10)
    if np.any(train_counts == 0) or np.any(heldout_counts == 0):
        raise ValueError(f"Fixed split has an empty client; no retry: train={train_counts}, heldout={heldout_counts}")
    train_ratings = full[train_idx][:, item_idx]
    heldout_ratings = full[heldout_idx][:, item_idx]
    train_means = np.stack([train_ratings[train_groups == client].mean(axis=0)
                            for client in range(10)])
    heldout_means = np.stack([heldout_ratings[heldout_groups == client].mean(axis=0)
                              for client in range(10)])
    item_mean = train_ratings.mean(axis=0)
    residual_ratings = train_ratings - item_mean[None, :]


    rating_svd = TruncatedSVD(n_components=9, n_iter=10, random_state=seed)
    rating_svd.fit(residual_ratings)
    loadings = rating_svd.components_.T
    raw_global = np.column_stack((item_mean, loadings[:, 5:9]))
    raw_local = loadings[:, :5]
    text_source = None
    global_definition = "Rating-only fallback: train item mean plus centered-train-rating SVD components 6-9"
    global_scope = "Training users only; no public-text features or heldout recommendation-arm scores"
    local_definition = "Same centered-train-rating SVD, components 1-5; no component overlaps the global block"
    xg_item = _normalize_block(raw_global, 0.25)
    xl_item = _normalize_block(raw_local, 0.75)
    slate_rng = np.random.default_rng(np.random.SeedSequence([seed, 904201]))
    slates = np.stack([slate_rng.choice(90, size=10, replace=False) for _ in range(100)])

    positive = np.stack([np.maximum(train_ratings[train_groups == client], 0.0).mean(axis=0)
                         for client in range(10)])
    slate_weights = positive[:, slates].mean(axis=2)
    row_sums = slate_weights.sum(axis=1, keepdims=True)
    preference = np.divide(slate_weights, row_sums,
                           out=np.full_like(slate_weights, 0.01), where=row_sums > 0.0)
    probabilities = (1.0 - slate_smoothing) * preference + slate_smoothing / 100.0
    noise_raw = float(np.sqrt(np.mean((train_ratings - train_means[train_groups]) ** 2)))
    if not noise_raw > 0.0:
        raise ValueError("Training residual RMS is zero; the unchanged config requires a positive proxy")
    xg, xl = xg_item[slates], xl_item[slates]
    pooled = probabilities.mean(axis=0)
    diagnostics = context_diagnostics_by_client(xg, xl, sampling_probabilities=probabilities,
                                                base_probabilities=pooled)
    selection_hasher = hashlib.sha256()
    _array_hash(selection_hasher, "complete_user_ids", complete_ids)
    manifest = {
        "schema": SCHEMA, "source": "Official UC Berkeley Jester Dataset 1, parts 1_1 and 1_2 only",
        "source_url": SOURCE_URL, "sources": [value[2] for value in loaded],
        "text_source": text_source, "feature_mode": "rating-fallback",
        "permission": "Freely available for research with requested Eigentaste citation; courtesy email optional",
        "pipeline_seed": seed, "train_fraction": train_fraction,
        "selection": "Only users with all 100 actual ratings present (99 means missing); no imputation",
        "selection_bias": "Complete-case population, not representative of all Jester users",
        "complete_user_ids_sha256": selection_hasher.hexdigest(),
        "users_original": sum(value[2]["rows_original"] for value in loaded),
        "users_complete": int(full.shape[0]), "users_train": int(cut),
        "users_heldout": int(full.shape[0] - cut),
        "train_client_counts": train_counts.tolist(), "heldout_client_counts": heldout_counts.tolist(),
        "heldout_min_client_count": int(heldout_counts.min()),
        "gauge_joke_ids": list(GAUGE_JOKE_IDS), "candidate_joke_ids": item_ids.tolist(),
        "grouping": "KMeans10/n_init10 fit on training gauge10 only; heldout assigned by their gauge10",
        "gauge_scope": "Historical preference features only; all gauge10 jokes excluded from every slate",
        "rating_transform": "actual rating / 10; no user demeaning, fitted reward, rescaling, or Gaussian augmentation",
        "global_features": global_definition,
        "global_feature_fit_scope": global_scope,
        "local_features": local_definition,
        "local_feature_fit_scope": "Training users only; heldout recommendation-arm scores never used",
        "svd_n_iter": 10, "context_norm": 1.0, "local_energy_fraction": 0.75,
        "feature_normalization": "Per-item separate row normalization: global energy .25, local .75",
        "feature_selection": "Training-rating features fixed before bandit evaluation; no regret-based selection or retry",
        "slate_seed_sequence": [seed, 904201], "slate_smoothing": slate_smoothing,
        "client_laws": "Mean max(train rating/10,0) by group/item; mean over each slate; normalize; 5% uniform smoothing",
        "client_law_fit_scope": "Training users and frozen slates only; no heldout rating inputs",
        "num_clients": 10, "num_actions": 10, "library_size": 100,
        "global_dim": 5, "local_dim": 5,
        "context_design_diagnostics": diagnostics.as_dict(),
        "context_design_by_client": [item.as_dict() for item in diagnostics.per_client],
        "item_slate_counts": np.bincount(slates.ravel(), minlength=90).tolist(),
        "train_noise_proxy": noise_raw,
        "noise_proxy_definition": "RMS of actual training rating/10 minus its training group/item mean, over all train users and candidate items",
        "noise_proxy_use": "Once-frozen operational calibration shared by every learner; no synthetic noise is drawn",
        "feedback": "IID heldout user row sampled with replacement in its client, independently of slate; reveal chosen real rating only",
        "oracle": "Heldout group/item empirical means; evaluate max within realized slate, not individual-user clairvoyant maximum",
        "theta_fields": "Structural zero placeholders only; no true theta or theta-recovery metric exists",
        "empirical_heterogeneity": None,
        "compatibility_heterogeneity": 0.3,
        "heterogeneity_semantics": "Empirical heterogeneity has no linear h value; .3 only occupies the existing config/trial interface, never rescales ratings or means",
        "theory_scope": "Empirical group means may be linearly misspecified; operational UCB bounds are not claimed valid under the paper's linear theorem",
        "uncertainty_scope": "Seeds vary IID row/context/algorithm tapes conditional on one frozen split/population; not across-user-generalization confidence intervals",
        "software": {"numpy": np.__version__, "scikit_learn": sklearn_version},
    }
    arrays = {
        "x_global": xg, "x_local": xl, "slate_item_ids": slates,
        "item_joke_ids": item_ids, "client_probabilities": probabilities,
        "train_ratings": train_ratings, "train_gauge": train_gauge,
        "train_groups": train_groups, "train_user_ids": complete_ids[train_idx],
        "heldout_ratings": heldout_ratings, "heldout_gauge": heldout_gauge,
        "heldout_groups": heldout_groups, "heldout_user_ids": complete_ids[heldout_idx],
        "train_group_means": train_means, "heldout_means": heldout_means,
        "gauge_centers": np.asarray(kmeans.cluster_centers_),
    }
    digest = _digest(arrays, manifest)
    manifest["bundle_digest"] = digest
    for array in arrays.values():
        array.setflags(write=False)
    return JesterBundle(**arrays, manifest=manifest, digest=digest)


def save_jester_bundle(bundle: JesterBundle, path: str | Path) -> Path:

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        previous = load_jester_bundle(destination)
        if previous.digest != bundle.digest:
            raise FileExistsError(f"Refusing to overwrite a different frozen bundle: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **{name: getattr(bundle, name) for name in _ARRAY_NAMES},
                                    manifest_json=np.asarray(json.dumps(bundle.manifest, sort_keys=True, allow_nan=False)))
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    sidecar = destination.with_suffix(".manifest.json")
    exported = dict(bundle.manifest, npz_sha256=file_sha256(destination))
    if sidecar.exists():
        current = json.loads(sidecar.read_text(encoding="utf-8"))
        if current != exported:
            raise FileExistsError(f"Refusing to overwrite an incompatible manifest: {sidecar}")
    else:
        temporary = sidecar.with_name(sidecar.name + ".tmp-" + uuid.uuid4().hex)
        try:
            temporary.write_text(json.dumps(exported, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(temporary, sidecar)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def load_jester_bundle(path: str | Path) -> JesterBundle:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in _ARRAY_NAMES}
        manifest = json.loads(str(archive["manifest_json"].item()))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unknown Jester empirical-bundle schema")
    digest = _digest(arrays, manifest)
    if digest != manifest.get("bundle_digest"):
        raise ValueError("Jester bundle digest does not match its arrays and manifest")
    if arrays["x_global"].shape != (100, 10, 5) or arrays["x_local"].shape != (100, 10, 5):
        raise ValueError("Frozen Jester feature dimensions do not match the requested protocol")
    if arrays["client_probabilities"].shape != (10, 100):
        raise ValueError("Frozen Jester client laws have invalid dimensions")
    if not np.allclose(arrays["client_probabilities"].sum(axis=1), 1.0) or np.any(arrays["client_probabilities"] <= 0):
        raise ValueError("Frozen Jester client probabilities are not positive normalized laws")
    if np.intersect1d(arrays["train_user_ids"], arrays["heldout_user_ids"]).size:
        raise ValueError("Training and heldout user IDs overlap")
    for client in range(10):
        rows = arrays["heldout_ratings"][arrays["heldout_groups"] == client]
        if rows.shape[0] == 0 or not np.array_equal(rows.mean(axis=0), arrays["heldout_means"][client]):
            raise ValueError("Frozen hidden empirical means do not match the heldout population")
    for array in arrays.values():
        array.setflags(write=False)
    return JesterBundle(**arrays, manifest=manifest, digest=digest)


@dataclass(frozen=True, kw_only=True)
class JesterRatingTrial(LinearBanditTrial):


    empirical_reward_tape: np.ndarray
    empirical_means: np.ndarray
    slate_item_ids: np.ndarray
    sampled_user_ids: np.ndarray
    bundle_digest: str

    @property
    def context_distribution(self) -> str:
        return "frozen Jester slates, train-only client laws, IID real heldout ratings with replacement; empirical oracle"

    def expected_rewards(self, client: int, time_index: int) -> np.ndarray:
        items = self.slate_item_ids[int(self.context_ids[client, time_index])]
        return self.empirical_means[client, items]

    def reward(self, client: int, time_index: int, action: int) -> float:
        return float(self.empirical_reward_tape[client, time_index, action])

    def pseudo_regret(self, client: int, time_index: int, action: int) -> float:
        means = self.expected_rewards(client, time_index)
        return float(np.max(means) - means[action])


def generate_jester_trial(bundle: JesterBundle, config, *, seed: int) -> JesterRatingTrial:

    expected = {"num_clients": 10, "num_actions": 10, "global_dim": 5,
                "local_dim": 5, "library_size": 100}
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(f"Jester requires fixed {name}={value}")
    n, horizon, arms = 10, int(config.horizon), 10
    pooled = bundle.client_probabilities.mean(axis=0)
    library_hasher = hashlib.sha256()
    for name, value in (("x_global", bundle.x_global), ("x_local", bundle.x_local), ("pooled", pooled)):
        _array_hash(library_hasher, name, value)
    library = PublicContextLibrary(
        x_global=bundle.x_global, x_local=bundle.x_local, context_norm=1.0,
        sampling_law="jester_real", sampling_probabilities=pooled,
        diagnostics=context_diagnostics(bundle.x_global, bundle.x_local, sampling_probabilities=pooled),
        digest=library_hasher.hexdigest(), local_energy_fraction=0.75,
    )
    profile_hasher = hashlib.sha256()
    _array_hash(profile_hasher, "client_probabilities", bundle.client_probabilities)
    profile = PublicClientContextProfile(
        base_probabilities=pooled, preference_probabilities=bundle.client_probabilities,
        sampling_probabilities=bundle.client_probabilities, mixture_eta=1.0,
        profile_strength=1.0, profile_seed=PIPELINE_SEED, digest=profile_hasher.hexdigest(),
        kind="jester", concentration=None,
    )
    context_rng = np.random.default_rng(np.random.SeedSequence([seed, 908117]))
    row_rng = np.random.default_rng(np.random.SeedSequence([seed, 610031]))
    forced_rng = np.random.default_rng(np.random.SeedSequence([seed, 610032]))
    context_uniforms = context_rng.random((n, horizon))
    context_ids = np.empty((n, horizon), dtype=np.int64)
    row_ids = np.empty((n, horizon), dtype=np.int64)
    cdf = np.cumsum(bundle.client_probabilities, axis=1)
    cdf[:, -1] = 1.0
    for client in range(n):
        context_ids[client] = np.searchsorted(cdf[client], context_uniforms[client], side="right")
        pool = np.flatnonzero(bundle.heldout_groups == client)
        row_ids[client] = pool[row_rng.integers(pool.size, size=horizon)]
    items = bundle.slate_item_ids[context_ids]
    rewards = bundle.heldout_ratings[row_ids[..., None], items]
    means = bundle.heldout_means[np.arange(n)[:, None, None], items]
    reward_noise = rewards - means
    exploration_uniforms = forced_rng.random((n, horizon))
    exploration_actions = forced_rng.integers(0, arms, size=(n, horizon), endpoint=False)
    theta = np.zeros(5)
    delta = np.zeros((n, 5))
    sampled_ids = bundle.heldout_user_ids[row_ids]
    hasher = hashlib.sha256()
    hasher.update(f"{SCHEMA};bundle={bundle.digest};seed={seed};horizon={horizon}".encode("ascii"))
    for name, value in (("contexts", context_ids), ("context_uniforms", context_uniforms),
                        ("sampled_rows", row_ids), ("real_rewards", rewards),
                        ("empirical_means", bundle.heldout_means),
                        ("forced_uniforms", exploration_uniforms), ("forced_actions", exploration_actions)):
        _array_hash(hasher, name, value)
    for array in (pooled, context_uniforms, context_ids, row_ids, rewards, reward_noise,
                  exploration_uniforms, exploration_actions, theta, delta, sampled_ids):
        array.setflags(write=False)
    return JesterRatingTrial(
        library=library, theta_global=theta, delta_local=delta, context_ids=context_ids,
        reward_noise=reward_noise, exploration_uniforms=exploration_uniforms,
        exploration_actions=exploration_actions, seed=int(seed), heterogeneity=0.3,
        digest=hasher.hexdigest(), client_context_profile=profile,
        client_context_diagnostics=context_diagnostics_by_client(
            bundle.x_global, bundle.x_local, sampling_probabilities=bundle.client_probabilities,
            base_probabilities=pooled), client_context_uniforms=context_uniforms,
        empirical_reward_tape=rewards, empirical_means=bundle.heldout_means,
        slate_item_ids=bundle.slate_item_ids, sampled_user_ids=sampled_ids, bundle_digest=bundle.digest,
    )
