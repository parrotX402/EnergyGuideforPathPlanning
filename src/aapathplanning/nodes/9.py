import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms

# ============================================================
# 0) Map B obstacles from your SDF (2D)
# ============================================================

# Oriented boxes: (cx, cy, L, W, yaw)
BOXES = [
    # Outer walls (5 x 0.15)
    (-2.425, 0.0, 5.0, 0.15,  1.5708),
    ( 2.425, 0.0, 5.0, 0.15, -1.5708),
    ( 0.0,  2.425,5.0, 0.15,  0.0),
    ( 0.0, -2.425,5.0, 0.15,  3.14159),

    # Chokes (1.8 x 0.15, yaw=90deg)
    (0.0,  1.45, 1.8, 0.15, 1.5708),
    (0.0, -1.45, 1.8, 0.15, 1.5708),

    # Trap
    (1.7,  1.0,  1.0, 0.15, 1.5708),
    (1.28, 1.5,  1.15,0.15, 0.0),
    (1.2,  0.5,  1.15,0.15, 0.0),

    # Zigzag
    (1.65, -0.6,  1.5, 0.15, 0.0),
    (0.65, -1.55, 1.5, 0.15, 0.0),
]

# Static circles: (cx, cy, r)
CIRCLES = [
    (-1.5,  1.0, 0.2),
    (-1.5, -1.0, 0.2),
]
# Dynamic obstacle ignored for repair constraint (optional for visualization)

# ============================================================
# 1) Geometry helpers: rotation + projection outside obstacles
# ============================================================

def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def project_outside_circle(p, cx, cy, r, margin):
    c = np.array([cx, cy], dtype=float)
    v = p - c
    d = np.linalg.norm(v)
    target = r + margin
    if d < 1e-12:
        return c + np.array([target, 0.0])
    if d < target:
        return c + v / d * target
    return p

def project_outside_oriented_box(p, cx, cy, L, W, yaw, margin):
    """
    Project p outside an oriented rectangle expanded by margin.
    If p is inside expanded rectangle, push it to the nearest face of expanded rectangle.
    """
    R = rot2(-yaw)
    pl = R @ (p - np.array([cx, cy], dtype=float))
    b = np.array([L/2 + margin, W/2 + margin], dtype=float)

    # already outside expanded box
    if (abs(pl[0]) >= b[0]) or (abs(pl[1]) >= b[1]):
        return p

    # inside: push to nearest face
    dx = b[0] - abs(pl[0])
    dy = b[1] - abs(pl[1])
    if dx < dy:
        pl[0] = np.sign(pl[0]) * b[0]
    else:
        pl[1] = np.sign(pl[1]) * b[1]

    return rot2(yaw) @ pl + np.array([cx, cy], dtype=float)

def project_outside_all(p, boxes, circles, margin):
    """
    Iterate projection a few times until point is outside all expanded obstacles.
    """
    x = p.copy()
    for _ in range(8):
        moved = False
        for (cx, cy, L, W, yaw) in boxes:
            x2 = project_outside_oriented_box(x, cx, cy, L, W, yaw, margin)
            if np.linalg.norm(x2 - x) > 1e-12:
                x = x2
                moved = True
        for (cx, cy, r) in circles:
            x2 = project_outside_circle(x, cx, cy, r, margin)
            if np.linalg.norm(x2 - x) > 1e-12:
                x = x2
                moved = True
        if not moved:
            break
    return x

# ============================================================
# 2) Trajectory smoothing: arc-length resampling (removes kinks)
# ============================================================

def resample_by_arclength(xy, N=None):
    xy = np.asarray(xy, float)
    if N is None:
        N = len(xy)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return xy.copy()
    t = np.linspace(0.0, s[-1], N)
    out = np.zeros((N, 2), float)
    j = 0
    for i, ti in enumerate(t):
        while j < len(s)-2 and s[j+1] < ti:
            j += 1
        a = (ti - s[j]) / max(s[j+1] - s[j], 1e-12)
        out[i] = (1-a)*xy[j] + a*xy[j+1]
    out[0] = xy[0]
    out[-1] = xy[-1]
    return out

# ============================================================
# 3) HARD repair: smooth update + hard projection each iter
#    Endpoints fixed strictly.
# ============================================================

def repair_trajectory_hard(xy, boxes, circles,
                           iters=500,
                           step=0.15,
                           w_smooth=0.9,
                           w_anchor=0.05,
                           margin=0.18,
                           resample_every=25):
    xy = np.asarray(xy, float)
    N = len(xy)
    if N < 3:
        return xy.copy()

    orig = xy.copy()
    out = xy.copy()

    for k in range(iters):
        dp = np.zeros_like(out)

        # synchronous Laplacian smoothing + anchor
        for i in range(1, N-1):
            p = out[i]
            lap = (out[i-1] + out[i+1] - 2.0*p)
            anchor = (orig[i] - p)
            dp[i] = w_smooth*lap + w_anchor*anchor

        out[1:-1] = out[1:-1] + step * dp[1:-1]

        # hard projection outside obstacles (expanded by margin)
        for i in range(1, N-1):
            out[i] = project_outside_all(out[i], boxes, circles, margin)

        # keep endpoints fixed
        out[0] = orig[0]
        out[-1] = orig[-1]

        # periodic resampling to remove kinks
        if resample_every and (k+1) % resample_every == 0:
            out = resample_by_arclength(out, N=N)
            out[0] = orig[0]
            out[-1] = orig[-1]

    return out

# ============================================================
# 4) Optional: collision check (after repair)
# ============================================================

def inside_oriented_box(px, py, cx, cy, L, W, yaw):
    dx, dy = px - cx, py - cy
    c, s = np.cos(-yaw), np.sin(-yaw)
    lx = c*dx - s*dy
    ly = s*dx + c*dy
    return (np.abs(lx) <= L/2) & (np.abs(ly) <= W/2)

def inside_circle(px, py, cx, cy, r):
    return (px-cx)**2 + (py-cy)**2 <= r**2

def collision_ratio(xy, boxes, circles, margin=0.0):
    """
    Count points inside obstacles expanded by margin.
    For boxes, expansion is handled by simply inflating (L,W) by 2*margin.
    """
    x = xy[:,0]
    y = xy[:,1]
    hit = np.zeros(len(x), dtype=bool)
    for (cx, cy, L, W, yaw) in boxes:
        hit |= inside_oriented_box(x, y, cx, cy, L+2*margin, W+2*margin, yaw)
    for (cx, cy, r) in circles:
        hit |= inside_circle(x, y, cx, cy, r+margin)
    return hit.mean(), hit.sum()

# ============================================================
# 5) Batch repair all npy -> *_repaired2.npy
# ============================================================

def batch_repair(pattern="path_*_pts*.npy"):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched: {pattern}")
        return []

    out_files = []
    print(f"Found {len(files)} files.")
    for fn in files:
        data = np.load(fn)
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"[SKIP] {fn}: not (N,2+)")
            continue

        xy = data[:, :2]
        repaired = repair_trajectory_hard(
            xy, BOXES, CIRCLES,
            iters=500,
            step=0.15,
            w_smooth=0.9,
            w_anchor=0.05,
            margin=0.18,
            resample_every=25
        )

        # Ensure endpoints exactly unchanged
        repaired[0]  = xy[0]
        repaired[-1] = xy[-1]

        out = data.copy()
        out[:,0] = repaired[:,0]
        out[:,1] = repaired[:,1]

        base, ext = os.path.splitext(fn)
        out_fn = base + "_repaired2" + ext
        np.save(out_fn, out)
        out_files.append(out_fn)

        cr, cnt = collision_ratio(repaired, BOXES, CIRCLES, margin=0.0)
        cr_m, cnt_m = collision_ratio(repaired, BOXES, CIRCLES, margin=0.18)
        print(f"[OK] {fn} -> {out_fn} | coll={cnt}/{len(repaired)} ({cr:.4f}), coll@margin={cnt_m}/{len(repaired)} ({cr_m:.4f})")

    return out_files

# ============================================================
# 6) Plot Map B + 6 repaired2 trajectories (your legend layout)
# ============================================================

def add_box_pose(ax, cx, cy, L, W, yaw, color, alpha=0.3, edgecolor=None, lw=1.0, zorder=0):
    rect = patches.Rectangle((cx - L/2, cy - W/2), L, W,
                             facecolor=color, alpha=alpha,
                             edgecolor=edgecolor if edgecolor else "none",
                             linewidth=lw, zorder=zorder)
    t = transforms.Affine2D().rotate_around(cx, cy, yaw) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

def plot_six_repaired2():
    # Use repaired2 files for your 6 curves
    traj6 = {
        ("TD3", "without"): "path_9_pts498_repaired2.npy",
        ("SAC", "without"): "path_1_pts461_repaired2.npy",
        ("GAP-SAC", "without"): "path_5_pts324_repaired2.npy",
        ("TD3", "with"): "path_3_pts461_repaired2.npy",
        ("SAC", "with"): "path_6_pts484_repaired2.npy",
        ("GAP-SAC", "with"): "path_0_pts205_repaired2.npy",
    }

    color_map = {"TD3": "tab:orange", "SAC": "tab:green", "GAP-SAC": "tab:blue"}
    ls_map = {"without": "--", "with": "-"}
    lw_map = {"without": 2.0, "with": 2.6}

    WALL_COLOR = "#C98C4A"

    fig, ax = plt.subplots(figsize=(9, 9))

    # draw map
    for (cx, cy, L, W, yaw) in BOXES:
        # color by category
        if abs(L-5.0) < 1e-6 and abs(W-0.15) < 1e-6:
            col, a, edge, lw = WALL_COLOR, 0.20, "black", 1.0
        elif abs(L-1.8) < 1e-6 and abs(W-0.15) < 1e-6:
            col, a, edge, lw = "green", 0.25, None, 0.0
        elif abs(W-0.15) < 1e-6 and (abs(L-1.15) < 1e-6 or abs(L-1.0) < 1e-6):
            col, a, edge, lw = "black", 0.25, None, 0.0
        elif abs(L-1.5) < 1e-6 and abs(W-0.15) < 1e-6:
            col, a, edge, lw = "orange", 0.25, None, 0.0
        else:
            col, a, edge, lw = "gray", 0.25, None, 0.0
        add_box_pose(ax, cx, cy, L, W, yaw, col, alpha=a, edgecolor=edge, lw=lw, zorder=0)

    for (cx, cy, r) in CIRCLES:
        ax.add_patch(plt.Circle((cx, cy), r, color="gray", alpha=0.20, zorder=1))

    # dynamic obstacle hint (optional)
    ax.add_patch(plt.Circle((-1.0, 0.0), 0.2, color="blue", alpha=0.15, zorder=1))
    ax.annotate("", xy=(-1.0, 1.5), xytext=(-1.0, -1.5),
                arrowprops=dict(arrowstyle="<->", lw=2, color="blue", alpha=0.15),
                zorder=1)

    # draw trajectories
    for (algo, eg_flag), fn in traj6.items():
        d = np.load(fn)
        x, y = d[:,0].astype(float), d[:,1].astype(float)
        label = f"{algo} ({'with EG' if eg_flag=='with' else 'without EG'})"
        ax.plot(x, y, color=color_map[algo], linestyle=ls_map[eg_flag],
                linewidth=lw_map[eg_flag], zorder=5, label=label)

    ax.set_title("Map B with HARD-Repaired Trajectories (with/without EG)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

    # legend 3x2
    desired_order = [
        "TD3 (with EG)", "GAP-SAC (with EG)", "SAC (with EG)",
        "TD3 (without EG)", "GAP-SAC (without EG)", "SAC (without EG)",
    ]
    handles, labels = ax.get_legend_handles_labels()
    hmap = {lab: h for h, lab in zip(handles, labels)}
    ax.legend([hmap[l] for l in desired_order if l in hmap],
              [l for l in desired_order if l in hmap],
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15),
              frameon=True, fontsize=10, columnspacing=1.2, handlelength=2.5)

    plt.tight_layout()
    plt.show()

# ============================================================
# 7) Run
# ============================================================

if __name__ == "__main__":
    # ① batch repair all paths -> repaired2
    batch_repair(pattern="path_*_pts*.npy")

    # ② plot 6 repaired2 on Map B
    plot_six_repaired2()