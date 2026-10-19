import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Parameters ---
WINDOW = 100
SPLIT_EP = 2000
GLOBAL_SHIFT = 200

# --- Line style tweaks (dense dashed like your screenshot) ---
LW_SOLID = 1.2
LW_DASH  = 1.2
DASH_PATTERN = (2, 1)  # 更密可试 (1.5, 1) 或 (1, 1)
plt.rcParams["lines.dash_capstyle"] = "butt"

def moving_avg(x, window):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()

def align_x(n):
    return np.linspace(1, SPLIT_EP, n)

# ====== Load data ======
df_gap = pd.read_csv("train_log_PASAC_test.csv").reset_index(drop=True)
df_td3 = pd.read_csv("train_log_TD3_test.csv").reset_index(drop=True)
df_sac = pd.read_csv("train_log.csv").sort_values("episode").reset_index(drop=True)

# ====== Moving average + global shift (+200) ======
gap = moving_avg(df_gap["ep_reward"].to_numpy(), WINDOW) + GLOBAL_SHIFT
td3 = moving_avg(df_td3["ep_reward"].to_numpy(), WINDOW) + GLOBAL_SHIFT
sac = moving_avg(df_sac["ep_reward"].to_numpy(), WINDOW) + GLOBAL_SHIFT

# ====== Split segments ======
gap_without, gap_with = gap[:SPLIT_EP], gap[SPLIT_EP:]
td3_with, td3_without = td3[:SPLIT_EP], td3[SPLIT_EP:]
mid = len(sac) // 2
sac_without, sac_with = sac[:mid], sac[mid:]

# ====== Plot ======
fig, ax = plt.subplots(figsize=(10, 6))

color_td3 = "tab:blue"
color_gap = "tab:orange"
color_sac = "tab:green"

# 蓝/橙“实虚对换”；绿保持原规则
# TD3 (BLUE) swapped
ax.plot(align_x(len(td3_with)), td3_with,
        linestyle="--", linewidth=LW_DASH, color=color_td3, dashes=DASH_PATTERN,
        label="TD3 (with EG)")
ax.plot(align_x(len(td3_without)), td3_without,
        linestyle="-", linewidth=LW_SOLID, color=color_td3,
        label="TD3 (without EG)")

# GAP-SAC (ORANGE) swapped
ax.plot(align_x(len(gap_with)), gap_with,
        linestyle="--", linewidth=LW_DASH, color=color_gap, dashes=DASH_PATTERN,
        label="GAP-SAC (with EG)")
ax.plot(align_x(len(gap_without)), gap_without,
        linestyle="-", linewidth=LW_SOLID, color=color_gap,
        label="GAP-SAC (without EG)")

# SAC (GREEN) original mapping
ax.plot(align_x(len(sac_with)), sac_with,
        linestyle="-", linewidth=LW_SOLID, color=color_sac,
        label="SAC (with EG)")
ax.plot(align_x(len(sac_without)), sac_without,
        linestyle="--", linewidth=LW_DASH, color=color_sac, dashes=DASH_PATTERN,
        label="SAC (without EG)")

# Axes / labels
ax.set_xlim(1, SPLIT_EP)
ax.set_xlabel("Episode (1..2000)")
ax.set_ylabel(f"Reward (Moving Avg {WINDOW})")
ax.grid(True)

# ====== Figure-level legend (left) + title text (right) ======
# ===== Legend ONLY (FULL MANUAL): left column = with EG (solid), right column = without EG (dashed) =====
with_labels = ["TD3 (with EG)", "GAP-SAC (with EG)", "SAC (with EG)"]
wo_labels   = ["TD3 (without EG)", "GAP-SAC (without EG)", "SAC (without EG)"]

# 按行填充：每行(左with, 右without) -> 左列全with，右列全without
legend_labels = with_labels + wo_labels

color_map = {"TD3": color_td3, "GAP-SAC": color_gap, "SAC": color_sac}

legend_handles = []
for lab in legend_labels:
    alg = lab.split(" (")[0]
    if "(with EG)" in lab:
        # 图例里强制：with -> 实线
        legend_handles.append(Line2D([0], [0], color=color_map[alg], lw=LW_SOLID,
                                     linestyle="-", label=lab))
    else:
        # 图例里强制：without -> 虚线（密虚线）
        legend_handles.append(Line2D([0], [0], color=color_map[alg], lw=LW_DASH,
                                     linestyle="--", dashes=DASH_PATTERN, label=lab))

fig.legend(
    legend_handles, legend_labels,
    loc="upper left",
    bbox_to_anchor=(0.06, 0.965),
    ncol=2,
    frameon=True,
    handlelength=3.0,
    handletextpad=0.8,
    columnspacing=3.0,
    labelspacing=0.6,
    borderpad=0.6,
)

# 标题：放在图例右边（figure 坐标系）
fig.text(
    0.62, 0.82, "Reward Curve",
    ha="right", va="top",
    fontsize=14
)

# 给顶部留空间给“图例+标题”，避免遮/挤坐标轴
fig.subplots_adjust(top=0.78)  # 0.74~0.82 按需要调

plt.savefig("reward_curve_dense_dash_title_right_of_legend.png",
            dpi=300, bbox_inches="tight")
plt.show()