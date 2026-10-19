能量引导路径规划：论文反向重建
================================

这是根据本地 IEEE 双栏论文和残留代码新建的复现工程。原文件均未修改。
目前交付的是**可运行的离线能量模型、采集接口、冻结奖励模块及验证**，
尚未完成 ROS/Gazebo 导航实验，也未复现论文成功率。

来源
----

* 论文 PDF：``/home/zhousiyuan/overleaf/Energy_Guided_Reward_Shaping_for_Robust_and_Generalizable_Deep_Reinforcement_Learning_in_Dynamic_Robot_Path_Planning.pdf``。
* 同标题的 LaTeX：``/home/zhousiyuan/overleaf/manuscript_source/root.tex``。
  方法见 156--223 行，能量网络参数表见 228--267 行，演示采集见 343--351 行。
* 原工程：``/home/zhousiyuan/overleaf/src/aapathplanning``。
* 本机已验证环境：Python 3.13、PyTorch 2.11.0 CPU、NumPy 2.2.6。
  现有 ``/opt/ros/humble`` 为 ROS 2；原工程依赖 ROS 1 的 ``rospy``，本机当前 Python 无此包。

已重建
------

* ``energyguide/model.py``：4→265→512→1024→512→256→1 能量网络；
  几何噪声 1、0.1、0.01；均匀抽噪声档位；带 sigma² 权重的 DSM；冻结推理。
* ``energyguide/cli.py``：Adam，学习率 1e-4，batch 64，默认 200001 次迭代，
  每 5000 次存检查点。保存配置、数据 SHA256、划分、归一化及训练日志。
* ``energyguide/data.py``：明确区分 paper4 和 legacy56；带结果标签时按成功/失败
  7:3 的采样概率采样；按 episode 分组留出验证集，避免同轨迹随机切分。
  不使用成功标签增加论文未描述的分类损失。
* ``energyguide/recording.py``：采集完整的 4 维特征和整条轨迹结果，保留碰撞和超时。
* ``energyguide/integration.py``：与策略算法无关的 r_total = r_env + alpha*r_eg 接口，
  从旧 ROS 状态排序读取局部几何。区分成功/碰撞的终止与时间上限的截断。
* ``energyguide/metrics.py``：所有试次作为成功率/碰撞率分母；成功步数只统计成功试次；
  没有成功样本时返回 null，不补造数值。

复现歧义与选择
--------------

1. **4 维论文与 56 维旧代码。** 论文为 [v, omega, heading, clearance]；
   原 ``main(copy).py`` 把两个 28 维状态拼接。旧状态只有 24 个激光距离、朝向误差、
   目标距离、最近障碍距离和索引，没有动作/速度。
   现存 204 份 ``near_dataset/*.npy`` 不能无损变成论文输入。
   ``legacy56`` 仅作旧代码对照分支，不能称为 4 维论文复现。

2. **DSM 符号。** 对 E=-log p，正确目标为
   ||grad E - (x_noisy-x_clean)/sigma²||²。
   论文写的是加号，最优解的符号对应 log p；若随后直接 exp(-E/beta)，
   奖励方向可能反转。默认 ``negative_log_density`` 使用减号；
   ``--convention paper_literal`` 保留论文加号以供对照。
   测试使用已知解析高斯势函数核对两种符号。

3. **能量零点与奖励范围。** DSM 只约束输入梯度，无法确定能量加性常数，
   普通 MLP 输出也未保证非负，所以论文宣称 exp(-E/beta)∈(0,1] 并不自动成立。
   默认重建假设：在最小噪声档，用训练集能量中位数为 reference，
   E_reward=softplus(E_density-reference)，再取 exp(-E_reward/beta)。
   ``FrozenReward(..., mapping="literal")`` 保留原始指数公式，允许返回 >1，
   溢出时报错。paper_literal 的 calibrated 模式会先反转模型输出的符号；
   literal 模式直接使用模型原始输出。校准不能证明导航有效。

4. **噪声条件、激活和归一化。** 论文显示 E(x,sigma)，却没有条件编码细节。
   默认给第一隐藏层加可学习噪声档位嵌入；``--unconditioned`` 可关闭。
   ELU 依据残留推理代码选取。输入使用训练集 mean/std 标准化（std 下限 1e-3），
   推理使用最小 sigma 档。以上均为明确的重建假设。

5. **失败样本。** 把成功/失败以 7:3 混合后做无条件 DSM，只能拟合混合数据分布，
   并不会自动学会把失败样本赋予更高能量。这里忠实保留混合采样，
   不凭空宣称已恢复正负样本鉴别机制。

6. **alpha 与 beta。** alpha=0.002 取自论文敏感性实验；beta 数值未给出，默认 1。
   与旧代码的 alpha=0.02、scale=500、线性裁剪奖励不同。
   终止步默认不加 shaping，继承旧代码；可设置 terminal_shaping=True 做对照。
   采集时优先同步的实测速度；默认使用物理动作命令作为速度近似，需记录该假设。

7. **场景运动。** 论文 x(t)=-1+0.36t 不具周期性，与“运动周期 20.94s”及
   原 ``move_slalom2.py`` 的有界往返轨迹不一致。当前不替论文选定唯一解释。
   地图 A 的可达采样区域、完整随机种子、策略训练预算和部分阈值也未完整交代。

运行与验证
----------

在此目录执行，无需下载依赖（本机已有 torch/numpy）::

    cd src/energyguide-reproduction
    python3 -m unittest discover -s tests -v
    python3 -m energyguide.cli smoke --out runs/new_synthetic_check --iterations 8

smoke 明确生成随机合成数据，只验证优化、存盘、加载和冻结奖励。
输出 ``verification.json`` 的状态为 ``synthetic_test_only``，不是导航实验结果。
每次使用新的输出目录，避免覆盖旧结果。

本地已保存两项检查（运行产物未上传，摘要见 VALIDATION.json）::

    runs/synthetic_smoke_01/verification.json
    runs/legacy_data_smoke_01/run.json

第一项使用合成 4 维输入，第二项对现存 56 维旧数据仅执行 8 次离线优化。
两项均不构成训练收敛或性能复现。

正式 4 维采集与训练
------------------

训练数据应为 NPZ，字段如下：

* ``features``：float32 [N,4]，固定列顺序 [v,omega,heading,clearance]；
  v∈[0,0.3] m/s，omega∈[-1,1] rad/s，heading∈[-pi,pi]，clearance≥0 m。
* ``success``：int [N]，该条转移所属轨迹最终成功为 1，碰撞/超时为 0。
* ``episode_id``：每条轨迹唯一的组标识；整条轨迹进入同一划分。
* ``map_id``：所有行必须为 A；采集器会写入 A，调用者必须实际在 Map A 采集。
* ``velocity_source``：由采集器记录 command/measured。

``DemonstrationRecorder`` 的用法见 ``examples/legacy_loop_snippet.py``。
``load_dataset`` 只会拒绝显式标注为 B 的数据；没有 map 元数据的旧数据来源
必须由操作者核实。单独 NPY 文件被视为一组，不能默认当成多条独立轨迹。
对于旧分片数据，按文件分组只能用于接口验证；若分片跨轨迹则不能视为独立验证集。

获得真实且有标签的 Map A 4 维演示后::

    python3 -m energyguide.cli pretrain \
      --data /absolute/path/to/demonstrations_map_a.npz \
      --out runs/map_a_paper4 \
      --representation paper4 --iterations 200001

旧数据只能显式使用兼容分支，无成功/失败标签时不能实现 7:3::

    python3 -m energyguide.cli pretrain \
      --data ../aapathplanning/nodes/near_dataset \
      --out runs/legacy56_new --representation legacy56 --allow-unlabelled --iterations 8

后续完整实验的执行条件
--------------------

* 恢复 ROS 1/Gazebo/TurtleBot3 仿真依赖及场景，核对论文地图 A/B 的实际对应关系。
* 在 Map A 用环境奖励训练/恢复 20000 轮 SAC 专家，并采集带物理动作的 4 维演示。
* 运行正式能量预训练；仅根据 Map A 校准与选参，Map B 不做演示补采/能量微调。
* 接入现存 SAC；补齐 TD3 并核对 GAP_SAC 原始算法，不能把残留 gated-attention SAC
  自动当成论文引用的完整 GAP_SAC。新模块的奖励接口本身不依赖这三种算法。
* 各策略骨干分别运行有/无能量引导的同条件多种子对照；保存逐回合记录、碰撞、
  超时、轨迹及训练日志，再计算 SR/CR/Steps/Length/Time 和明确阈值的平滑度指标。

本工程没有启动 ROS、Gazebo、真实机器人，也没有运行 200001 次正式训练或导航评估。
完整复现的缺口是数据、仿真环境及实验执行，不应拿本次 smoke 检查替代。
