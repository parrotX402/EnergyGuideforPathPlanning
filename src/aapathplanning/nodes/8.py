import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms

# =======================
# 1) Trajectories (as-is)
# =======================
TRAJ_FILES = {
    ("TD3", "without"): "path_0_pts205_egopt.npy",
    ("SAC", "without"): "path_1_pts461_egopt.npy",
    ("GAP-SAC", "without"): "path_2_pts437_egopt.npy",
    ("TD3", "with"): "path_4_pts401_egopt.npy",
    ("SAC", "with"): "path_6_pts484.npy",
    ("GAP-SAC", "with"): "path_0_pts205.npy",
}

# =======================
# 2) Map B geometry (WORLD coords, no transform)
# =======================
WORLD_BOXES = [
    # outer walls
    (-2.425, 0.0, 5.0, 0.15,  1.5708),
    ( 2.425, 0.0, 5.0, 0.15, -1.5708),
    ( 0.0,  2.425, 5.0, 0.15,  0.0),
    ( 0.0, -2.425, 5.0, 0.15,  3.14159),

    # choke
    (0.0,  1.45, 1.8, 0.15, 1.5708),
    (0.0, -1.45, 1.8, 0.15, 1.5708),

    # trap
    (1.7,  1.0,  1.0, 0.15, 1.5708),
    (1.28, 1.5,  1.15, 0.15, 0.0),
    (1.2,  0.5,  1.15, 0.15, 0.0),

    # zigzag
    (1.65, -0.6,  1.5, 0.15, 0.0),
    (0.65, -1.55, 1.5, 0.15, 0.0),
]

WORLD_CIRCLES = [
    (-1.5,  1.0, 0.2),
    (-1.5, -1.0, 0.2),
]

WORLD_INNER = (-2.35, 2.35, -2.35, 2.35)  # xmin,xmax,ymin,ymax

# =======================
# 3) Plot style
# =======================
color_map = {"TD3": "tab:orange", "SAC": "tab:green", "GAP-SAC": "tab:blue"}
ls_map = {"without": "--", "with": "-"}
lw_map = {"without": 2.0, "with": 2.6}

def draw_box(ax, cx, cy, L, W, yaw, color, alpha=0.25, z=0, edgecolor=None, lw=0.0):
    rect = patches.Rectangle(
        (cx - L/2, cy - W/2), L, W,
        facecolor=color, alpha=alpha,
        edgecolor=edgecolor if edgecolor else "none",
        linewidth=lw, zorder=z
    )
    t = transforms.Affine2D().rotate_around(cx, cy, yaw) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

# =======================
# 4) Draw
# =======================
fig, ax = plt.subplots(figsize=(9, 9))

# ---- all boxes -> wood color; circles stay gray ----
WOOD = "#C98C4A"
WOOD_ALPHA = 0.22

for (cx, cy, L, W, yaw) in WORLD_BOXES:
    is_wall = (abs(L - 5.0) < 1e-6 and abs(W - 0.15) < 1e-6)
    draw_box(
        ax, cx, cy, L, W, yaw,
        color=WOOD,
        alpha=WOOD_ALPHA,
        z=0,
        edgecolor="black" if is_wall else None,  # outer walls have outline
        lw=1.0 if is_wall else 0.0
    )

for (cx, cy, r) in WORLD_CIRCLES:
    ax.add_patch(plt.Circle((cx, cy), r, color="gray", alpha=0.20, zorder=1))

# ---- inner free-space bounds (optional debug; comment out if not needed) ----
xmin, xmax, ymin, ymax = WORLD_INNER
ax.add_patch(
    patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        fill=False, edgecolor="gray", linewidth=1.5, linestyle=":", zorder=2
    )
)

# ---- trajectories ----
for (algo, eg_flag), fn in TRAJ_FILES.items():
    d = np.load(fn)
    if d.ndim != 2 or d.shape[1] < 2:
        raise ValueError(f"{fn} must be (N,2+) array. Got {d.shape}")
    x, y = d[:, 0].astype(float), d[:, 1].astype(float)
    label = f"{algo} ({'with EG' if eg_flag=='with' else 'without EG'})"
    ax.plot(
        x, y,
        color=color_map[algo],
        linestyle=ls_map[eg_flag],
        linewidth=lw_map[eg_flag],
        zorder=5,
        label=label
    )

# ---- legend order (3 columns x 2 rows) ----
desired_order = [
    "TD3 (with EG)", "GAP-SAC (with EG)", "SAC (with EG)",
    "TD3 (without EG)", "GAP-SAC (without EG)", "SAC (without EG)",
]
handles, labels = ax.get_legend_handles_labels()
hmap = {lab: h for h, lab in zip(handles, labels)}
ax.legend(
    [hmap[l] for l in desired_order if l in hmap],
    [l for l in desired_order if l in hmap],
    ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18),
    frameon=True, fontsize=10, columnspacing=1.2, handlelength=2.5
)

ax.set_title("Map B (WORLD) + Trajectories (as-is)", fontsize=14, fontweight="bold")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, linestyle="--", alpha=0.25)

# ---- view window: include both map + traj ----
traj_arrays = [np.load(f) for f in TRAJ_FILES.values()]
allx = np.concatenate([d[:, 0] for d in traj_arrays])
ally = np.concatenate([d[:, 1] for d in traj_arrays])
pad = 0.6
ax.set_xlim(min(allx.min(), xmin) - pad, max(allx.max(), xmax) + pad)
ax.set_ylim(min(ally.min(), ymin) - pad, max(ally.max(), ymax) + pad)

plt.tight_layout()
plt.show()