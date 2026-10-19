# Energy-Guided Robot Path Planning

本仓库汇集路径规划原工程，以及依据论文重建的离线能量模型与奖励模块。

对应论文：**Energy-Guided Reward Shaping for Robust and Generalizable Deep Reinforcement Learning in Dynamic Robot Path Planning**。

```text
src/
├── aapathplanning/             # 原 ROS 1 / Gazebo / TurtleBot3 工程
└── energyguide-reproduction/   # 新建的独立 PyTorch 能量模块
```

## 当前实现状态

- 原工程包含 SAC 相关实现、环境、地图、启动文件和现存数据。
- 新工程实现 4 维能量网络、多尺度 DSM、冻结奖励、演示采集接口和评估统计接口。
- 11 项单元测试通过；合成 4 维数据和现存 56 维数据各执行了 8 次离线优化检查。
- **尚未完成正式能量预训练及 ROS/Gazebo 导航性能复现。** 奖励接口可供不同策略骨干调用，但这不代表三个骨干的完整实验已经完成。

论文描述的核心机制是离线学习能量模型，在策略训练时冻结，组合环境奖励与能量奖励。论文中的 4 维输入与旧代码的 56 维输入、DSM 符号、能量零点和障碍运动公式存在待核对之处。具体差异、默认修正和复现假设见 [重建说明](src/energyguide-reproduction/README.rst)。

## 独立能量模块

需要 Python 3.10+、PyTorch 2.0+、NumPy 1.24+。本次验证使用 Python 3.13、PyTorch 2.11.0 CPU 和 NumPy 2.2.6；未验证最低版本组合。

```bash
git clone https://github.com/parrotX402/EnergyGuideforPathPlanning.git
cd EnergyGuideforPathPlanning
python -m pip install -e ./src/energyguide-reproduction
cd src/energyguide-reproduction
python -m unittest discover -s tests -v
python -m energyguide.cli smoke --out runs/synthetic_check --iterations 8
```

`smoke` 仅使用明确标注的合成数据验证优化、存盘、加载和冻结推理，不产生导航性能结论。每次使用新的输出目录。

现存数据只能用旧代码的 56 维兼容分支做检查：

```bash
python -m energyguide.cli pretrain \
  --data ../aapathplanning/nodes/near_dataset \
  --out runs/legacy56_check --representation legacy56 \
  --allow-unlabelled --iterations 8
```

正式 4 维复现需要从 Map A 重新采集带物理速度、朝向误差、障碍距离和轨迹结果的数据。旧的 56 维状态对没有速度，不能直接当成论文 4 维演示。数据格式及完整预训练命令见重建说明。

## ROS 工程

`src/aapathplanning` 保留原代码、文件名及数据，不自动改写旧绝对路径。运行前需要 ROS 1（原目标为 Ubuntu 20.04 / Noetic）、Gazebo 和 TurtleBot3，并修正环境路径和缺失依赖。当前本地系统只有 ROS 2 Humble，尚未验证该 ROS 1 工程启动。

缺失的原 NEAR 能量检查点未恢复；不能把现存策略权重或新生成的测试检查点当作它。工作空间顶层原 `CMakeLists.txt` 是指向旧 Noetic 安装的 Windows 链接序列化文件，未上传；请在正确 ROS 1 环境中初始化 Catkin 工作空间。

## 上传内容

上传源码、仿真场景、原有 `.npy` 数据及图表，保留现有 MIT 许可证和原文件中的作者署名。虚拟环境、缓存、IDE 配置、82 个模型权重文件及本次运行产物保留在本地，未上传。

文件清单和 SHA256 见 [UPLOAD_MANIFEST.json](UPLOAD_MANIFEST.json)；验证摘要见 [VALIDATION.json](src/energyguide-reproduction/VALIDATION.json)。重建说明中的本地绝对路径用于标明来源，在其他机器上需替换。
