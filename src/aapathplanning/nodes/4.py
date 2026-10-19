import numpy as np
import matplotlib.pyplot as plt

# 1. 设置文件和目标点 (根据你的实际需求修改)
file_to_plot = 'path_8_pts492.npy'
fixed_goals = [[-1.0, 1.0], [2.0, -2.0], [1.0, 1.0]] 

# 2. 加载轨迹
data = np.load(file_to_plot)
x, y = data[:, 0], data[:, 1]

plt.figure(figsize=(8, 8))

# 3. 画出地图围栏 (TurtleBot3 默认地图通常是 4x4)
wall = 2.0
plt.plot([-wall, wall, wall, -wall, -wall], [-wall, -wall, wall, wall, -wall], 
         'k-', linewidth=2, label='Static Obstacle')

# 4. 画出固定目标点 (论文常用星星标注)
goals_arr = np.array(fixed_goals)
plt.scatter(goals_arr[:, 0], goals_arr[:, 1], c='gold', s=200, 
            marker='*', edgecolors='black', label='Target Goals', zorder=5)

# 5. 画出机器人轨迹
plt.plot(x, y, color='#1f77b4', linewidth=2, label='Path 8 (Successful)')

# 6. 标注起点和终点
plt.scatter(x[0], y[0], color='green', s=100, label='Start')
plt.scatter(x[-1], y[-1], color='red', marker='X', s=100, label='End')

# 细节美化
plt.title("Robot Navigation Trajectory (Episode 8)", fontsize=14)
plt.xlabel("Global X [m]")
plt.ylabel("Global Y [m]")
plt.axis('equal')
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()

plt.savefig('final_report_plot.png', dpi=300)
print(f"分析完成！图片已保存为 final_report_plot.png")
plt.show()