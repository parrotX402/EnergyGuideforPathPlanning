import numpy as np
# --- 关键修改：告诉 Matplotlib 不要使用交互式窗口 ---
import matplotlib
matplotlib.use('Agg') 
# ----------------------------------------------
import matplotlib.pyplot as plt
import os

# 1. 加载文件
file_path = 'turtlebot_traj_test.npy' 
if not os.path.exists(file_path):
    print(f"找不到文件: {file_path}")
else:
    data = np.load(file_path)
    x = data[:, 0]
    y = data[:, 1]

    # 2. 绘图
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, label='Trajectory', color='blue', linewidth=1)
    plt.scatter(x[0], y[0], color='green', label='Start') # 起点
    plt.scatter(x[-1], y[-1], color='red', marker='X', label='End') # 终点

    plt.title("Robot Path")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    # 3. 保存
    output_img = "trajectory_result.png"
    plt.savefig(output_img)
    print(f"成功！图片已保存到当前目录下的: {output_img}")
    # 注意：这里删除了 plt.show()，因为它会导致程序卡住