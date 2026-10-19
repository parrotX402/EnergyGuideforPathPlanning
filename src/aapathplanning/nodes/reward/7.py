import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置全局参数：字体和学术配色
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
COLORS = ['#0C5DA5', '#00B945', '#FF9500'] # 经典学术深蓝、翠绿、琥珀橙

def smooth(data, weight=0.85):
    """指数平滑，模拟 Tensorboard 的平滑效果"""
    last = data[0]
    smoothed = []
    for point in data:
        if np.isnan(point):
            smoothed.append(last)
            continue
        last = last * weight + (1 - weight) * point
        smoothed.append(last)
    return smoothed

# 1. 加载数据
df = pd.read_csv('train_log_PASAC_test.csv')
x = df['episode'].values

# 2. 创建画布 (双栏论文通常建议单图宽度 3.3 英寸)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# --- 左图：Reward (主指标) ---
raw_reward = df['ep_reward'].values
smooth_reward = smooth(raw_reward, weight=0.9)
ax1.plot(x, raw_reward, color=COLORS[0], alpha=0.2, lw=1) # 原始数据浅色
ax1.plot(x, smooth_reward, color=COLORS[0], label='EG-PASAC (Ours)', lw=2) # 平滑曲线深色

ax1.set_title('Training Convergence', fontsize=12, fontweight='bold')
ax1.set_xlabel('Episodes', fontsize=10)
ax1.set_ylabel('Cumulative Reward', fontsize=10)
ax1.legend(loc='lower right', frameon=False)
ax1.grid(True, ls='--', alpha=0.4)

# --- 右图：Success Rate & Near Mean (双坐标或双线) ---
success_rate = df['success'].rolling(window=30, min_periods=1).mean().values
ax2.plot(x, success_rate, color=COLORS[1], lw=2, label='Success Rate')

# 共享 X 轴绘制 near_mean，体现 EG 信号的稳定性
ax2_twin = ax2.twinx()
near_smooth = smooth(df['near_mean'].values, weight=0.95)
ax2_twin.plot(x, near_smooth, color=COLORS[2], ls='--', lw=1.5, label='EG Signal (Near Mean)')

ax2.set_title('Task Success & Guidance Signal', fontsize=12, fontweight='bold')
ax2.set_xlabel('Episodes', fontsize=10)
ax2.set_ylabel('Success Rate', fontsize=10)
ax2_twin.set_ylabel('Energy-Guided Signal', fontsize=10, color=COLORS[2])

# 合并图例
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower right', frameon=False, fontsize=8)

ax2.grid(True, ls='--', alpha=0.4)

# 3. 细节优化：去除上方和右方边框
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False) # 右侧坐标轴保留 ax2 的右边框

plt.tight_layout()
plt.savefig('Publication_Result.pdf', bbox_inches='tight', dpi=600) # 保存为矢量图 PDF
plt.savefig('Publication_Result.png', bbox_inches='tight', dpi=300)
print("论文级图表已生成: Publication_Result.pdf")