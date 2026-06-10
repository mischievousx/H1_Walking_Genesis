# 🤖 H1 Walking RL Training

Reinforcement learning training for the Unitree H1 humanoid robot's walking gait, using the Genesis physics simulator and a custom PPO implementation. Includes evaluation/plotting tooling, a ROS2 deployment package, and a written report.

## ⚙️ Requirements

- NVIDIA GPU with CUDA 12.x (tested on RTX 5090 32GB)
- Conda

## 🛠️ Environment Setup

### 1. Create conda environment

```bash
conda create -n unitree-genesis python=3.10 -y
conda activate unitree-genesis
```

### 2. Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Genesis

```bash
pip install genesis-world==1.0.0
```

### 4. Install remaining dependencies

```bash
pip install tensorboard imageio
```

## 🏃 Training

Basic training with default parameters:

```bash
python -u train.py
```

Logs are printed every 10 updates. TensorBoard:

```bash
tensorboard --logdir runs
```

`train.py` parses hyperparameter overrides directly from `sys.argv` (no `argparse`). Supported flags include `--resume`, `--resume-from`, `--save-dir`, `--rollout`, `--ent-coef`, and per-component reward scale overrides: `--dof-acc`, `--contact-no-vel`, `--action-rate`, `--ang-vel-xy`, `--feet-swing-height`, `--tracking-lin-vel`.

### 🔬 Experiment: dof_acc ablation

Three parallel runs comparing joint acceleration penalty strength. All other parameters identical.

**Baseline (`dof_acc = -1e-6`)**
```bash
python -u train.py --save-dir checkpoints_v4_r48s --rollout 48 \
    --ang-vel-xy -0.2 --feet-swing-height -40.0 --tracking-lin-vel 2.0 \
    --dof-acc -1e-6 --contact-no-vel -0.5 --action-rate -0.03
```

**5x (`dof_acc = -5e-6`)**
```bash
python -u train.py --save-dir checkpoints_v4_r48s_acc5x --rollout 48 \
    --ang-vel-xy -0.2 --feet-swing-height -40.0 --tracking-lin-vel 2.0 \
    --dof-acc -5e-6 --contact-no-vel -0.5 --action-rate -0.03
```

**10x (`dof_acc = -1e-5`)**
```bash
python -u train.py --save-dir checkpoints_v4_r48s_acc10x --rollout 48 \
    --ang-vel-xy -0.2 --feet-swing-height -40.0 --tracking-lin-vel 2.0 \
    --dof-acc -1e-5 --contact-no-vel -0.5 --action-rate -0.03
```

Key reward scale changes vs. defaults:

| Parameter | Default | This experiment |
|-----------|---------|-----------------|
| `tracking_lin_vel` | 1.0 | 2.0 |
| `ang_vel_xy` | -0.05 | -0.2 |
| `feet_swing_height` | -20.0 | -40.0 |
| `contact_no_vel` | -0.2 | -0.5 |
| `action_rate` | -0.01 | -0.03 |
| `dof_acc` | -2.5e-7 | -1e-6 / -5e-6 / -1e-5 |

Across the evaluation tooling below, `acc5x` (`dof_acc = -5e-6`) comes out as the best-balanced checkpoint: lowest `dof_acc`/`dof_jerk`, best velocity tracking pooled over 0.3/0.6/0.9 m/s, and the best disturbance recovery. See `report/report.tex` for the full analysis.

## 🎬 Visualization

### `play.py` — single straight-line clip

```bash
python play.py checkpoints/checkpoints_v4_r48s_acc5x/h1_walk_best.pt
```

| Flag | Default | Description |
|------|---------|-------------|
| `checkpoint` | (required) | Path to a `.pt` checkpoint |
| `--out` | `videos/play.mp4` | Output video path |
| `--steps` | `300` | Simulation steps to record |
| `--vx` | `0.5` | Commanded forward velocity (m/s) |
| `--vy` | `0.0` | Commanded lateral velocity (m/s) |
| `--yaw` | `0.0` | Commanded yaw rate (rad/s) |
| `--cpu` | off | Use CPU backend |

### `play_ros2_path.py` — ROS2 closed-loop route demo

Closed-loop architecture: the Genesis sim publishes `/imu`, a `DemoSequenceNode` converts heading error to `/cmd_vel` via P-control, and the sim feeds `/cmd_vel` back into the RL policy. The robot follows a fixed deterministic 6-segment route (forward → left turn → forward → right turn → forward → stop, 1250 steps total).

```bash
source /opt/ros/humble/setup.bash
python3 play_ros2_path.py \
    --checkpoint checkpoints/checkpoints_v4_r48s_acc5x/h1_walk_best.pt \
    --out videos/ros2_path_closed_acc5x.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `checkpoints/checkpoints_v4_r48s_acc5x/h1_walk_best.pt` | Policy checkpoint to run |
| `--out` | `videos/ros2_path_closed_acc5x.mp4` | Output video path |
| `--cpu` | off | Use CPU backend |

### `demo_sequence.py` — multi-phase demo (no ROS2 required)

Same phase sequence as above (forward → left turn → forward → right turn → forward → stop), driven directly without ROS2 topics.

```bash
python demo_sequence.py --checkpoint checkpoints/checkpoints_v4_r48s_acc5x/h1_walk_best.pt --out videos/demo_sequence.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `checkpoints_v5_003/h1_walk_best.pt` (stale, override it) | Policy checkpoint to run |
| `--out` | `videos/demo_sequence.mp4` | Output video path |
| `--cpu` | off | Use CPU backend |

## 📊 Evaluation (`eval/`)

### `eval_breakdown.py` — reward component breakdown

Pins straight-line velocity commands at `vx ∈ {0.3, 0.6, 0.9}` m/s and reports the mean raw value of each reward component (`tracking_lin_vel`, `dof_acc`, `action_rate`, `ang_vel_xy`, `feet_swing_height`, `contact_no_vel`) plus the total clipped reward.

```bash
python eval/eval_breakdown.py --checkpoint checkpoints/checkpoints_v4_r48s_acc5x/h1_walk_best.pt
python eval/eval_breakdown.py --compare   # all three v4_r48s ablation checkpoints
```

### `eval_smoothness.py` — joint kinematic smoothness

Same evaluation harness and speed range (`vx ∈ {0.3, 0.6, 0.9}` m/s), reporting `dof_vel_jitter` (std of joint velocity deltas) and `dof_jerk` (mean |joint acceleration delta / dt|).

```bash
python eval/eval_smoothness.py --checkpoint checkpoints/checkpoints_v4_r48s_acc5x/h1_walk_best.pt
python eval/eval_smoothness.py --compare
```

### `eval_robustness_traj.py` — push-recovery COM trajectories

Applies lateral push impulses at several magnitudes (fractions of `MAX_PUSH_VEL`) to a 256-env batch per checkpoint and plots center-of-mass height/lateral-displacement recovery trajectories to `report/figs/com_traj_<checkpoint>.png`.

```bash
python eval/eval_robustness_traj.py
```

## 📈 Plots (`plots/`)

All scripts write figures to `report/figs/`.

| Script | Output | Source data |
|--------|--------|-------------|
| `plot_training_curves.py` | `training_curves.png` — reward & value-loss vs. PPO update | `logs/train_v4_r48s*.log` |
| `plot_ros2_paths.py` | 2D top-down path comparison | `*_path.npy` recorded by `play_ros2_path.py` |
| `make_gait_strip.py` | Side-by-side gait-cycle comparison | `videos/videos_v4_r48s*/play_best.mp4` |
| `make_ros2_strip.py` | 2x3 key-frame strip of the ROS2 route | `videos/ros2_path/r48s.mp4` |

```bash
python plots/plot_training_curves.py
```

## 🚀 ROS2 Deployment (`ros2_h1_deploy/`)

`ament_python` ROS2 package (depends on `rclpy`, `geometry_msgs`, `std_msgs`) that wraps a trained policy as a deployable node, mirroring the closed-loop architecture used by `play_ros2_path.py`.

## 📄 Report (`report/`)

`report/report.tex` is the course report ("基于深度强化学习的H1人形机器人行走控制系统设计"), with figures in `report/figs/`. Compile with `xelatex` (uses `ctex` for Chinese text).

## 📁 Project Structure

```
├── train.py            # Training entry point
├── env.py              # H1WalkingEnv (Genesis + reward shaping)
├── ppo.py              # PPO + ActorCritic implementation
├── play.py             # Single-clip policy visualization → video
├── play_ros2_path.py   # ROS2 closed-loop fixed-route demo → video
├── demo_sequence.py    # Multi-phase demo without ROS2
├── eval/                # Evaluation scripts (reward breakdown, smoothness, robustness)
├── plots/               # Plotting scripts → report/figs/
├── report/              # LaTeX course report + figures
├── ros2_h1_deploy/      # ROS2 ament_python deployment package
├── logs/                # Training & evaluation logs
├── scripts/             # Helper shell scripts (run/shutdown)
├── robot/urdf/          # H1 URDF model
├── checkpoints/         # Saved policy checkpoints
├── runs/                # TensorBoard logs
└── videos/              # Rendered videos
```
