import os
import math
import numpy as np

from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# =========================
# 1) Map B obstacles
# =========================
WALL_COLOR = "#C98C4A"
BOXES = [
    (-2.425, 0.0, 5.0, 0.15,  1.5708, WALL_COLOR, 0.20),
    ( 2.425, 0.0, 5.0, 0.15, -1.5708, WALL_COLOR, 0.20),
    ( 0.0,  2.425,5.0, 0.15,  0.0,    WALL_COLOR, 0.20),
    ( 0.0, -2.425,5.0, 0.15,  3.14159,WALL_COLOR, 0.20),

    (0.0,  1.45, 1.8, 0.15, 1.5708, "green", 0.25),
    (0.0, -1.45, 1.8, 0.15, 1.5708, "green", 0.25),

    (1.7,  1.0,  1.0, 0.15, 1.5708, "black", 0.25),
    (1.28, 1.5,  1.15,0.15, 0.0,    "black", 0.25),
    (1.2,  0.5,  1.15,0.15, 0.0,    "black", 0.25),

    (1.65, -0.6,  1.5, 0.15, 0.0, "orange", 0.25),
    (0.65, -1.55, 1.5, 0.15, 0.0, "orange", 0.25),
]
CIRCLES = [
    (-1.5,  1.0, 0.2, "gray", 0.20),
    (-1.5, -1.0, 0.2, "gray", 0.20),
]

RECTS = [(cx, cy, L, W, yaw) for (cx, cy, L, W, yaw, _, _) in BOXES]
CIRCS = [(cx, cy, r) for (cx, cy, r, _, _) in CIRCLES]

# =========================
# 2) Geometry helpers
# =========================
def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)

def rect_sdf_and_closest(p: np.ndarray, cx, cy, L, W, yaw):
    p = p.astype(float)
    R = rot2(-yaw)
    pl = R @ (p - np.array([cx, cy], dtype=float))

    hx, hy = L / 2.0, W / 2.0
    q = np.array([abs(pl[0]) - hx, abs(pl[1]) - hy], dtype=float)
    outside = np.maximum(q, 0.0)
    outside_dist = float(np.linalg.norm(outside))
    inside_dist = float(min(max(q[0], q[1]), 0.0))
    sdf = outside_dist + inside_dist

    cl = np.array([np.clip(pl[0], -hx, hx), np.clip(pl[1], -hy, hy)], dtype=float)

    if q[0] < 0 and q[1] < 0:
        penx = hx - abs(pl[0])
        peny = hy - abs(pl[1])
        if penx < peny:
            cl[0] = math.copysign(hx, pl[0] if pl[0] != 0 else 1.0)
            cl[1] = pl[1]
        else:
            cl[1] = math.copysign(hy, pl[1] if pl[1] != 0 else 1.0)
            cl[0] = pl[0]

    Rw = rot2(yaw)
    closest_world = (Rw @ cl) + np.array([cx, cy], dtype=float)
    return sdf, closest_world

def circle_sdf(p: np.ndarray, cx, cy, r):
    p = p.astype(float)
    return float(np.linalg.norm(p - np.array([cx, cy], dtype=float)) - r)

def min_sdf_and_closest(p: np.ndarray):
    best_sdf = float("inf")
    best_closest = None

    for (cx, cy, L, W, yaw) in RECTS:
        sdf, closest = rect_sdf_and_closest(p, cx, cy, L, W, yaw)
        if sdf < best_sdf:
            best_sdf = sdf
            best_closest = closest

    for (cx, cy, r) in CIRCS:
        sdf = circle_sdf(p, cx, cy, r)
        if sdf < best_sdf:
            best_sdf = sdf
            v = p - np.array([cx, cy], dtype=float)
            n = np.linalg.norm(v)
            if n < 1e-9:
                best_closest = np.array([cx + r, cy], dtype=float)
            else:
                best_closest = np.array([cx, cy], dtype=float) + v / n * r

    return best_sdf, best_closest

def count_inside(xy: np.ndarray) -> int:
    cnt = 0
    for p in xy:
        sdf, _ = min_sdf_and_closest(p)
        if sdf < 0:
            cnt += 1
    return cnt

# ✅ 只修“在障碍里”的点：不会把轨迹搞成折线
def post_fix_inside_only(xy: np.ndarray, margin=0.02, iters=60) -> np.ndarray:
    p = xy.astype(float).copy()
    for _ in range(iters):
        changed = False
        for i in range(len(p)):
            sdf, closest = min_sdf_and_closest(p[i])
            if sdf < 0:
                v = p[i] - closest
                n = float(np.linalg.norm(v))
                if n < 1e-9:
                    v = p[i] - (0.5 * (p[i] + closest))
                    n = float(np.linalg.norm(v)) + 1e-9
                p[i] = closest - (v / n) * margin
                changed = True
        if not changed:
            break
    return p.astype(np.float32)

# =========================
# 3) Path parameterization & control points
# =========================
def arclength_param(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-12:
        return np.linspace(0.0, 1.0, len(xy))
    return s / s[-1]

def pick_control_points(xy: np.ndarray, s: np.ndarray, K: int):
    s_targets = np.linspace(0.0, 1.0, K)
    idx = np.searchsorted(s, s_targets)
    idx = np.clip(idx, 0, len(s) - 1)
    for i in range(1, len(idx)):
        if idx[i] <= idx[i - 1]:
            idx[i] = min(idx[i - 1] + 1, len(s) - 1)
    s_ctrl = s[idx]
    p_ctrl = xy[idx].astype(float)
    return idx, s_ctrl, p_ctrl

def spline_from_control(s_ctrl: np.ndarray, p_ctrl: np.ndarray):
    csx = CubicSpline(s_ctrl, p_ctrl[:, 0], bc_type="natural")
    csy = CubicSpline(s_ctrl, p_ctrl[:, 1], bc_type="natural")
    return csx, csy

# =========================
# 4) Objective
# =========================
def softplus(x, beta=10.0):
    return np.log1p(np.exp(beta * x)) / beta

def optimize_path(
    xy: np.ndarray,
    K: int = 35,
    sample_M: int = 300,
    margin: float = 0.05,
    w_fit: float = 2.0,
    w_smooth: float = 0.8,
    w_obs: float = 120.0,
    w_anchor: float = 50.0,
    maxiter: int = 1000,
):
    xy = xy.astype(float)
    s = arclength_param(xy)
    idx, s_ctrl, p0 = pick_control_points(xy, s, K)

    p_start = p0[0].copy()
    p_end = p0[-1].copy()

    s_fit = np.linspace(0.0, 1.0, min(len(xy), 500))
    x_fit = np.interp(s_fit, s, xy[:, 0])
    y_fit = np.interp(s_fit, s, xy[:, 1])
    xy_fit = np.stack([x_fit, y_fit], axis=1)

    s_obs = np.linspace(0.0, 1.0, sample_M)

    def unpack(z): return z.reshape(K, 2)
    def pack(P): return P.reshape(-1)

    z0 = pack(p0)

    def loss(z):
        P = unpack(z)
        csx, csy = spline_from_control(s_ctrl, P)

        pred_fit = np.stack([csx(s_fit), csy(s_fit)], axis=1)
        fit_term = np.mean(np.sum((pred_fit - xy_fit) ** 2, axis=1))

        dd = P[:-2] - 2 * P[1:-1] + P[2:]
        smooth_term = np.mean(np.sum(dd ** 2, axis=1)) if len(dd) > 0 else 0.0

        pred_obs = np.stack([csx(s_obs), csy(s_obs)], axis=1)
        obs_pen = 0.0
        for p in pred_obs:
            sdf, _ = min_sdf_and_closest(p)
            obs_pen += softplus(margin - sdf, beta=20.0) ** 2
        obs_term = obs_pen / len(pred_obs)

        anchor_term = np.sum((P[0] - p_start) ** 2) + np.sum((P[-1] - p_end) ** 2)

        return w_fit * fit_term + w_smooth * smooth_term + w_obs * obs_term + w_anchor * anchor_term

    res = minimize(
        loss,
        z0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10},
    )

    P_opt = unpack(res.x)
    csx, csy = spline_from_control(s_ctrl, P_opt)
    xy_new = np.stack([csx(s), csy(s)], axis=1)
    return xy_new.astype(np.float32), res

# =========================
# 5) Batch run (✅换这里：加入你新文件)
# =========================
if __name__ == "__main__":
    input_files = [
        "path_0_pts205.npy",
        "path_1_pts461.npy",
        "path_2_pts437.npy",
        "path_3_pts461.npy",
        "path_4_pts401.npy",
        "path_5_pts324.npy",
        "path_6_pts484.npy",
        "path_7_pts275.npy",
        "path_8_pts492.npy",
        "path_9_pts498.npy",
        "path_10_pts101.npy",
    ]

    # 这组参数更偏“避障”，保形也还不错
    K = 35
    margin = 0.05
    w_obs = 120.0
    w_fit = 2.0
    w_smooth = 0.8
    w_anchor = 50.0
    sample_M = 300
    maxiter = 1000

    for fn in input_files:
        if not os.path.exists(fn):
            print(f"[skip] missing: {fn}")
            continue

        arr = np.load(fn)
        xy = arr[:, :2].astype(float)
        inside_before = count_inside(xy)

        xy_opt, info = optimize_path(
            xy,
            K=K,
            sample_M=sample_M,
            margin=margin,
            w_fit=w_fit,
            w_smooth=w_smooth,
            w_obs=w_obs,
            w_anchor=w_anchor,
            maxiter=maxiter,
        )

        inside_mid = count_inside(xy_opt)

        # ✅轻量“只修内点”，通常把 inside 清到 0
        xy_opt2 = post_fix_inside_only(xy_opt, margin=0.02, iters=80)
        inside_after = count_inside(xy_opt2)

        out = arr.astype(np.float32).copy()
        out[:, :2] = xy_opt2

        out_fn = fn.replace(".npy", "_egopt.npy")
        np.save(out_fn, out)
        print(
            f"[ok] {fn} -> {out_fn} | inside: {inside_before} -> {inside_mid} -> {inside_after} | "
            f"iters={info.nit} success={info.success}"
        )