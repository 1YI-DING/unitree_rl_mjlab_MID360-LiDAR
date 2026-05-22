# Unitree RL Mjlab — MID360 LiDAR 感知运动控制

基于 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 扩展，在原有速度跟踪与动作模仿功能基础上，新增了面向 **Unitree MID360 LiDAR** 的感知运动控制训练方案，实现机器人在复杂地形（楼梯、坡道）下的 Sim2Real 部署。

<div align="center">

| 仿真（感知行走）| 实机（爬台阶）|
|:---:|:---:|
| <img src="doc/gif/g1-velocity.gif" width="300"/> | <img src="doc/gif/g1-velocity-real.gif" width="300"/> |

</div>

---

## 主要特性

- **MID360 感知运动控制**：基于激光雷达高度图的感知策略，机器人在仿真中使用理想高度扫描训练，通过渐进稀疏化课程学习过渡到真实 MID360 稀疏点云
- **高度图记忆机制**：补偿 LiDAR 脚下盲区，将历史扫描点随机器人前进自动平移，填充当前帧缺失区域
- **三阶段训练流程**：盲走预训练 → 感知粗糙地形 v1/v2 → 爬台阶精调，每阶段可加载前阶段 checkpoint 继续训练
- **完整 Sim2Real 链路**：支持 MuJoCo 仿真验证 → unitree_mujoco 硬件接口仿真 → 实机部署

支持机器人：**Go2、A2、As2、G1（29DOF）、G1-23DOF、H1\_2、H2、R1**

---

## 安装

### 环境要求

- Ubuntu 22.04
- NVIDIA GPU（驱动 ≥ 550）
- Python 3.11（推荐使用 Conda）

详细安装步骤请参考 [doc/setup_zh.md](doc/setup_zh.md)。

### 快速开始

```bash
# 克隆仓库（含 submodule）
git clone --recurse-submodules https://github.com/1YI-DING/unitree_rl_mjlab_MID360-LiDAR.git
cd unitree_rl_mjlab_MID360-LiDAR

# 创建并激活环境
conda create -n unitree_rl_mjlab python=3.11
conda activate unitree_rl_mjlab

# 安装依赖
pip install -e .
```

---

## 使用流程

```
训练  →  仿真验证  →  仿真部署  →  实机部署
```

---

## 1. 速度跟踪训练

### 1.1 盲走（无感知）

适用于所有机器人的基础速度跟踪训练：

```bash
# 平地
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096

# 粗糙地形
python scripts/train.py Unitree-G1-Rough --env.scene.num-envs=4096
```

多 GPU 训练：

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

**全部可用任务：**

| 机器人 | 平地 | 粗糙地形 |
|--------|------|----------|
| G1（29DOF） | `Unitree-G1-Flat` | `Unitree-G1-Rough` |
| G1-23DOF | `Unitree-G1-23Dof-Flat` | `Unitree-G1-23Dof-Rough` |
| Go2 | `Unitree-Go2-Flat` | `Unitree-Go2-Rough` |
| H1\_2 | `Unitree-H1_2-Flat` | `Unitree-H1_2-Rough` |
| H2 | `Unitree-H2-Flat` | `Unitree-H2-Rough` |
| A2 | `Unitree-A2-Flat` | `Unitree-A2-Rough` |
| As2 | `Unitree-As2-Flat` | `Unitree-As2-Rough` |
| R1 | `Unitree-R1-Flat` | `Unitree-R1-Rough` |

### 1.2 MID360 感知训练（G1 专用）

感知训练分三个递进阶段，**每阶段可从上一阶段的 checkpoint 继续训练**：

#### 阶段一：感知粗糙地形 v1

初步引入高度图观测，地形以楼梯为主（台阶高度 4–16 cm）：

```bash
python scripts/train.py Unitree-G1-Perceptive-Rough-v1 \
  --env.scene.num-envs=4096 \
  --agent.resume=True
```

#### 阶段二：感知粗糙地形 v2（推荐主训练）

核心 Sim2Real 阶段：使用 `bridged_mid360_height_scan_memory` 观测函数，在训练过程中渐进式地：
- 将理想密集高度图过渡到 MID360 前向稀疏扫描（x 范围 0.25–1.15 m）
- 引入随机稀疏化（保留率 65%–100%）和帧丢失（2% 概率）
- 利用历史记忆补全脚下盲区，记忆有效期 0.8 s

```bash
python scripts/train.py Unitree-G1-Perceptive-Rough-v2 \
  --env.scene.num-envs=4096 \
  --agent.resume=True
```

#### 阶段三：爬台阶精调（可选）

在 v2 基础上，针对上台阶行为精调（台阶占比 60%），新增了落脚点前移、踏板边缘惩罚等专项奖励：

```bash
python scripts/train.py Unitree-G1-Perceptive-Rough-v2-StepUp-Finetune \
  --env.scene.num-envs=4096 \
  --agent.resume=True
```

**其他感知相关任务：**

| 任务 ID | 说明 |
|---------|------|
| `Unitree-G1-Perceptive-Rough-v2-Straight` | 仅直行命令，抑制偏航漂移 |
| `Unitree-G1-Perceptive-Rough-v2-Step14-Up` | 固定 14 cm 台阶测试 |
| `Unitree-G1-Rough-Step14` / `...-Step14-Down` | 固定 14 cm 台阶上行 / 下行 |

**训练结果保存路径：**`logs/rsl_rl/<experiment_name>/<date_time>/model_<iter>.pt`

---

## 2. 动作模仿训练

### 2.1 准备动作文件

将 CSV 格式动作文件转换为训练用的 NPZ 格式：

```bash
python scripts/csv_to_npz.py \
  --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz \
  --input-fps 30 \
  --output-fps 50 \
  --robot g1   # g1 或 g1_23dof
```

### 2.2 训练

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs=4096
```

可用任务：`Unitree-G1-Tracking-No-State-Estimation`、`Unitree-G1-23Dof-Tracking-No-State-Estimation`

---

## 3. 仿真验证

```bash
# 速度跟踪
python scripts/play.py Unitree-G1-Perceptive-Rough-v2 \
  --checkpoint_file=logs/rsl_rl/g1_perceptive_rough_v2/<date>/model_<iter>.pt

# 动作模仿
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz \
  --checkpoint_file=logs/rsl_rl/g1_tracking/<date>/model_<iter>.pt
```

训练每次保存模型时会同步导出 `policy.onnx`，可直接用于实机部署。

---

## 4. 实机部署

### 4.1 依赖安装

```bash
# cyclonedds（DDS 通信中间件）
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git
# unitree_sdk2 已作为 submodule 集成，无需单独安装
```

### 4.2 启动机器人

将机器人**吊装启动**，等待进入`零力矩模式`，然后按遥控器 `L2 + R2` 进入`调试模式`（关节阻尼状态）。

### 4.3 连接机器人

网线连接机器人网口，配置 PC 网络：
- IP：`192.168.123.222`
- 子网掩码：`255.255.255.0`

使用 `ifconfig` 记录网卡名称（如 `enp5s0`）。

### 4.4 编译

将 `policy.onnx` 放入 `deploy/robots/g1/config/policy/velocity/v0/exported/`，然后：

```bash
cd deploy/robots/g1
mkdir build && cd build
cmake .. && make
```

> 无感知版部署代码位于 `deploy-no perception/`，结构相同，不依赖 MID360。

### 4.5 仿真预验证（推荐）

实机部署前先用 unitree_mujoco 验证：

```bash
# 编译仿真器
cd simulate && mkdir build && cd build
cmake .. && make -j8

# 启动仿真器（需连接手柄）
./simulate/build/unitree_mujoco

# 另开终端，启动控制程序
cd deploy/robots/g1/build
./g1_ctrl --network=lo
```

### 4.6 实机运行

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=enp5s0   # 替换为实际网卡名
```

---

## 常用训练参数

| 参数 | 说明 |
|------|------|
| `--env.scene.num-envs` | 并行环境数量，越大越快，需 GPU 内存支持 |
| `--gpu-ids` | 多 GPU 训练，如 `--gpu-ids 0 1` |
| `--agent.resume` | 从上次 checkpoint 继续训练 |
| `--agent.seed` | 随机种子，用于复现 |
| `--env.rewards` | 奖励函数参数覆盖 |

---

## 项目结构

```
unitree_rl_mjlab_MID360-LiDAR/
├── src/
│   ├── tasks/velocity/         # 速度跟踪任务（含感知任务）
│   │   ├── config/g1/          # G1 感知任务配置（v1/v2/stepup）
│   │   └── mdp/observations.py # MID360 高度图观测实现
│   ├── tasks/tracking/         # 动作模仿任务
│   └── assets/                 # 机器人 MJCF 模型、动作文件
├── deploy/                     # 含感知的 C++ 部署代码
├── deploy-no perception/       # 无感知版本部署代码（对照）
├── simulate/                   # unitree_mujoco 仿真部署
├── scripts/                    # 训练/推理入口脚本
├── unitree_ros2/               # Unitree ROS2 通信（submodule）
└── unitree_sdk2/               # Unitree SDK2（submodule）
```

---

## 致谢

- [mjlab](https://github.com/mujocolab/mjlab)：训练与运行框架基础
- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)：本项目的上游仓库
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)：PPO 强化学习算法实现
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp)：GPU 加速仿真接口
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)：动作模仿框架
