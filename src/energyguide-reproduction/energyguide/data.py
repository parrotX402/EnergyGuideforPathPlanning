"""Read explicitly identified features. Never infer missing v/omega from 56D states."""

from pathlib import Path
import hashlib

import numpy as np


def feature4(v, omega, heading, clearance):
    x = np.asarray([v, omega, heading, clearance], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite")
    if not 0 <= v <= 0.3 or not -1 <= omega <= 1 or clearance < 0:
        raise ValueError("Expected v in [0,0.3], omega in [-1,1], clearance >= 0")
    x[2] = (heading + np.pi) % (2 * np.pi) - np.pi
    return x


def legacy56(state, next_state):
    """Use raw old 28D states. Their last angle index is normalized by 23 once."""
    s = np.asarray(state, dtype=np.float32).copy()
    sp = np.asarray(next_state, dtype=np.float32).copy()
    if s.shape != (28,) or sp.shape != (28,):
        raise ValueError("Legacy mode requires two 28D raw states")
    s[-1] /= 23
    sp[-1] /= 23
    return np.concatenate([s, sp])


def load_dataset(path, representation, allow_unlabelled=False):
    """NPZ: features, success (0/1), optional episode_id, map_id.

    NPY shards: features only. A shard is treated as one split group; this is
    conservative only if shards really are independent trajectories/sessions.
    """
    path = Path(path)
    paths = sorted(path.glob("*.npy")) if path.is_dir() else [path]
    if not paths:
        raise ValueError("No input data found")
    dim = {"paper4": 4, "legacy56": 56}[representation]
    features, labels, groups, sources = [], [], [], []
    for i, p in enumerate(paths):
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as data:
                x = np.array(data["features"], dtype=np.float32)
                y = np.array(data["success"]) if "success" in data else None
                g = np.array(data["episode_id"]).astype(str) if "episode_id" in data else np.full(len(x), str(i))
                if "map_id" in data and np.any(data["map_id"].astype(str) != "A"):
                    raise ValueError("Offline energy training requires Map A only")
        elif p.suffix == ".npy":
            x = np.asarray(np.load(p, allow_pickle=False), dtype=np.float32)
            y, g = None, np.full(len(x), str(i))
        else:
            raise ValueError("Expected a .npz, .npy, or directory of .npy shards")
        if x.ndim != 2 or x.shape[1] != dim or len(x) == 0:
            raise ValueError(f"{p}: expected nonempty [N,{dim}], got {x.shape}. "
                             "56D state pairs do not contain the velocities required for paper4.")
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite features in {p}")
        if representation == "paper4":
            if (np.any(x[:, 0] < 0) or np.any(x[:, 0] > 0.300001)
                    or np.any(np.abs(x[:, 1]) > 1.000001)
                    or np.any(np.abs(x[:, 2]) > np.pi + 1e-5) or np.any(x[:, 3] < 0)):
                raise ValueError("paper4 order/range: [v 0..0.3, omega -1..1, heading -pi..pi, clearance >=0]")
        if y is None:
            if not allow_unlabelled:
                raise ValueError("Missing success labels; 7:3 resampling is impossible. "
                                 "Use --allow-unlabelled only for an explicitly labelled reconstruction variant.")
            y = np.full(len(x), -1, dtype=np.int64)
        if y.shape != (len(x),) or g.shape != (len(x),):
            raise ValueError("success and episode_id must have length N")
        if not np.isin(y, [-1, 0, 1] if allow_unlabelled else [0, 1]).all():
            raise ValueError("success must contain 0/1")
        # One trajectory cannot be both successful and failed.
        for group in np.unique(g):
            if len(np.unique(y[g == group])) != 1:
                raise ValueError("Each episode_id must have one success label")
        features.append(x)
        labels.append(y.astype(np.int64))
        groups.append(np.array([f"{i}:{v}" for v in g]))
        sources.append({"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "shape": list(x.shape)})
    return np.concatenate(features), np.concatenate(labels), np.concatenate(groups), sources


def group_split(groups, seed=0, holdout=0.2):
    if not 0 <= holdout < 1:
        raise ValueError("holdout must be in [0,1)")
    unique = np.unique(groups)
    if holdout == 0:
        return np.arange(len(groups)), np.array([], dtype=np.int64)
    if len(unique) < 2:
        raise ValueError("Need >=2 independent episode/shard groups for holdout; or explicitly --holdout 0")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = max(1, min(len(unique) - 1, round(len(unique) * holdout)))
    held = np.isin(groups, unique[:n])
    return np.flatnonzero(~held), np.flatnonzero(held)


def sample_indices(labels, count, rng):
    """7:3 in expectation; these labels do NOT add a discriminative loss."""
    if np.all(labels == -1):
        return rng.integers(len(labels), size=count)
    if np.any(labels == -1):
        raise ValueError("Do not mix labelled and unlabelled data")
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not len(pos) or not len(neg):
        raise ValueError("Training split needs both outcomes for the paper's 7:3 sampling")
    success = rng.random(count) < 0.7
    result = np.empty(count, dtype=np.int64)
    result[success] = rng.choice(pos, success.sum())
    result[~success] = rng.choice(neg, (~success).sum())
    return result
