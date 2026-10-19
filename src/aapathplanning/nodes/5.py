import numpy as np
import matplotlib.pyplot as plt

# 1. 加载提取出的路径
data = np.load('path_1_pts461.npy')
dt = 0.1 # 假设你的采样时间间隔是 0.1s

# 2. 计算速度 (v = ds/dt)
diff = np.diff(data, axis=0)
v = np.sqrt(np.sum(diff**2, axis=1)) / dt

# 3. 计算近似功率 (Energy Cost)
# 这里模拟一个简化的控制代价模型
power = v**2  # 瞬时消耗

# 4. 计算累积能量 (Cumulative Energy)
cumulative_energy = np.cumsum(power * dt)

# 5. 开始画图
fig, ax1 = plt.subplots(figsize=(10, 5))

# 绘制瞬时能量
color = 'tab:blue'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Instantaneous Cost ($v^2$)', color=color)
ax1.plot(v**2, color=color, alpha=0.6, label='Instantaneous')
ax1.tick_params(axis='y', labelcolor=color)

# 创建双坐标轴绘制累积能量
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Cumulative Energy', color=color)
ax2.plot(cumulative_energy, color=color, linewidth=2, label='Cumulative')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Energy Transformation and Consumption Analysis')
fig.tight_layout()
plt.grid(True, linestyle=':', alpha=0.5)
plt.savefig('energy_curve.png', dpi=300)
print("能量变换曲线已生成：energy_curve.png")
plt.show()