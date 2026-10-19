import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 加载原始数据
file_name = 'turtlebot_traj_test.npy'
if not os.path.exists(file_name):
    print(f"找不到文件: {file_name}")
    exit()

data = np.load(file_name)

# 2. 自动切分逻辑
# 计算点与点之间的欧式距离
diff = np.diff(data, axis=0)
dist = np.sqrt(np.sum(diff**2, axis=1))

# 阈值设为 0.5米（如果瞬间移动超过0.5米，判定为Reset重置）
threshold = 0.5 
split_indices = np.where(dist > threshold)[0] + 1
trajectories = np.split(data, split_indices)

print(f"--- 数据分析报告 ---")
print(f"总点数: {len(data)}")
print(f"检测到路径数量: {len(trajectories)}")
# --- 之前的切分逻辑保持不变 ---

print(f"--- 批量提取开始 ---")

# 使用循环遍历所有检测到的路径
for i, t in enumerate(trajectories):
    # 构建文件名，例如: path_0_pts205.npy
    save_name = f'path_{i}_pts{len(t)}.npy'
    
    # 保存每一条路径
    np.save(save_name, t)
    
    print(f"已保存路径 [{i}]: {save_name} (点数: {len(t)})")

print(f"--- 全部提取完成，共保存 {len(trajectories)} 个文件 ---")

# 可视化：一次性画出所有路径，用不同颜色区分
plt.figure(figsize=(10, 8))
for i, t in enumerate(trajectories):
    plt.plot(t[:,0], t[:,1], label=f'Path {i}')

plt.title("All Detected Episodes")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis('equal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 标签放在图外，防止遮挡
plt.grid(True)
plt.tight_layout()
plt.show()