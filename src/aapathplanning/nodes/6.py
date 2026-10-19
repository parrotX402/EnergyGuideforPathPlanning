import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms

# ===== 6 trajectories (REPAIRED) =====
TRAJ_FILES = {
    ("TD3", "without"): "path_9_pts498_repaired.npy",
    ("SAC", "without"): "path_1_pts461_repaired.npy",
    ("GAP-SAC", "without"): "path_5_pts324_repaired.npy",
    ("TD3", "with"): "path_3_pts461_repaired.npy",
    ("SAC", "with"): "path_6_pts484_repaired.npy",
    ("GAP-SAC", "with"): "path_0_pts205_repaired.npy",
}

# ===== Style =====
color_map = {"TD3": "tab:orange", "SAC": "tab:green", "GAP-SAC": "tab:blue"}
ls_map = {"without": "--", "with": "-"}
lw_map = {"without": 2.0, "with": 2.6}

# ===== Map B obstacles (from your SDF) =====
WALL_COLOR = "#C98C4A"  # wood-ish
BOXES = [
    # outer walls
    (-2.425, 0.0, 5.0, 0.15,  1.5708, WALL_COLOR, 0.20),
    ( 2.425, 0.0, 5.0, 0.15, -1.5708, WALL_COLOR, 0.20),
    ( 0.0,  2.425,5.0, 0.15,  0.0,    WALL_COLOR, 0.20),
    ( 0.0, -2.425,5.0, 0.15,  3.14159,WALL_COLOR, 0.20),

    # choke (green)
    (0.0,  1.45, 1.8, 0.15, 1.5708, "green", 0.25),
    (0.0, -1.45, 1.8, 0.15, 1.5708, "green", 0.25),

    # trap (black)
    (1.7,  1.0,  1.0, 0.15, 1.5708, "black", 0.25),
    (1.28, 1.5,  1.15,0.15, 0.0,    "black", 0.25),
    (1.2,  0.5,  1.15,0.15, 0.0,    "black", 0.25),

    # zigzag (orange)
    (1.65, -0.6,  1.5, 0.15, 0.0, "orange", 0.25),
    (0.65, -1.55, 1.5, 0.15, 0.0, "orange", 0.25),
]
CIRCLES = [
    (-1.5,  1.0, 0.2, "gray", 0.20),
    (-1.5, -1.0, 0.2, "gray", 0.20),
]
# dynamic obstacle hint (optional)
DYN_CIRCLE = (-1.0, 0.0, 0.2, "blue", 0.18)  # t=0 position
DYN_RANGE_Y = (-1.5, 1.5)                    # waypoint swing range

def add_box_pose(ax, cx, cy, L, W, yaw, color, alpha=0.3, edgecolor=None, lw=1.0, zorder=0):
    rect = patches.Rectangle((cx - L/2, cy - W/2), L, W,
                             facecolor=color, alpha=alpha,
                             edgecolor=edgecolor if edgecolor else "none",
                             linewidth=lw, zorder=zorder)
    t = transforms.Affine2D().rotate_around(cx, cy, yaw) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

# ===== Plot =====
fig, ax = plt.subplots(figsize=(9, 9))

# 1) draw map first
for (cx, cy, L, W, yaw, col, a) in BOXES:
    edge = "black" if col == WALL_COLOR else None
    lw = 1.0 if col == WALL_COLOR else 0.0
    add_box_pose(ax, cx, cy, L, W, yaw, col, alpha=a, edgecolor=edge, lw=lw, zorder=0)

for (cx, cy, r, col, a) in CIRCLES:
    ax.add_patch(plt.Circle((cx, cy), r, color=col, alpha=a, zorder=1))

# dynamic obstacle (optional visual cue)
cx, cy, r, col, a = DYN_CIRCLE
ax.add_patch(plt.Circle((cx, cy), r, color=col, alpha=a, zorder=1))
ax.annotate(
    "", xy=(cx, DYN_RANGE_Y[1]), xytext=(cx, DYN_RANGE_Y[0]),
    arrowprops=dict(arrowstyle="<->", lw=2, color="blue", alpha=0.18),
    zorder=1
)

# 2) draw trajectories
for (algo, eg_flag), fn in TRAJ_FILES.items():
    data = np.load(fn)
    x, y = data[:, 0].astype(float), data[:, 1].astype(float)
    label = f"{algo} ({'with EG' if eg_flag=='with' else 'without EG'})"
    ax.plot(x, y,
            color=color_map[algo],
            linestyle=ls_map[eg_flag],
            linewidth=lw_map[eg_flag],
            zorder=5,
            label=label)

# cosmetics
ax.set_title("Map B with Repaired Trajectories (with/without EG)", fontsize=14, fontweight="bold")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, linestyle="--", alpha=0.25)

# limits: include both map and traj
allx = np.concatenate([np.load(f)[:, 0] for f in TRAJ_FILES.values()])
ally = np.concatenate([np.load(f)[:, 1] for f in TRAJ_FILES.values()])
pad = 0.6
ax.set_xlim(min(allx.min(), -2.7) - pad, max(allx.max(),  2.7) + pad)
ax.set_ylim(min(ally.min(), -2.7) - pad, max(ally.max(),  2.7) + pad)

# legend order 3x2
desired_order = [
    "TD3 (with EG)", "GAP-SAC (with EG)", "SAC (with EG)",
    "TD3 (without EG)", "GAP-SAC (without EG)", "SAC (without EG)",
]
handles, labels = ax.get_legend_handles_labels()
hmap = {lab: h for h, lab in zip(handles, labels)}
ax.legend(
    [hmap[l] for l in desired_order if l in hmap],
    [l for l in desired_order if l in hmap],
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    frameon=True,
    fontsize=10,
    columnspacing=1.2,
    handlelength=2.5,
)

plt.tight_layout()
plt.show()