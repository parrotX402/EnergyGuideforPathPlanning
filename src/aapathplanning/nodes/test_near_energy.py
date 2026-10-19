import numpy as np
from near_infer import NearEnergyModel

# 1) near checkpoint 路径
CKPT_PATH = "/home/zhosuiyuan/桌面/near/isaacgymenvs/learning/motion_ncsn/run/logs/near_run1/checkpoint.pth"

# 2) 加载 near 模型
near = NearEnergyModel(CKPT_PATH, device="cpu")
print("[OK] Near model loaded")

# 3) 随便加载一个 near 数据（你已经有）
X = np.load(
    "/home/zhosuiyuan/桌面/src/aapathplanning/nodes/near_dataset/near_X_0100.npy"
)

print("dataset shape:", X.shape)  # (N,56) 或 (N,56?)

# 4) 随机取一个样本
x56 = X[np.random.randint(len(X))]
print("sample x56 shape:", x56.shape)

# 5) 计算 energy
E = near.energy(x56)
R = near.shaped_reward(x56)

print(f"energy = {E:.6f}")
print(f"shaped_reward = {R:.6f}")

