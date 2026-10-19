import numpy as np
import glob
import pandas as pd

def analyze_one(filename):
    data = np.load(filename)

    # 计算里程 (Path Length)
    dist = np.sqrt(np.sum(np.diff(data, axis=0)**2, axis=1))
    total_dist = np.sum(dist)

    # 计算直线位移 (Displacement)
    displacement = np.sqrt(np.sum((data[-1] - data[0])**2))

    # 计算效率 (直线距离 / 实际路径)
    efficiency = displacement / total_dist if total_dist > 0 else 0

    return {
        "文件": filename,
        "采样点数": len(data),
        "行驶里程(m)": total_dist,
        "任务位移(m)": displacement,
        "路径平滑效率(%)": efficiency * 100
    }

# 1) 批量读取：按你的文件命名规则
files = sorted(glob.glob("path_*.npy"))

# 2) 逐个计算并汇总
rows = [analyze_one(f) for f in files]
df = pd.DataFrame(rows)

# 3) 排序（可选）：按效率从高到低
df = df.sort_values("路径平滑效率(%)", ascending=False).reset_index(drop=True)

# 4) 打印列表（表格）
print(df.to_string(index=False, formatters={
    "行驶里程(m)": "{:.3f}".format,
    "任务位移(m)": "{:.3f}".format,
    "路径平滑效率(%)": "{:.2f}".format,
}))

# 5) 可选：保存为 CSV
df.to_csv("path_metrics_list.csv", index=False, encoding="utf-8-sig")
print("\n已保存: path_metrics_list.csv")
