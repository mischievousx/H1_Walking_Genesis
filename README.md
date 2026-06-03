# 🤖 H1 Walking RL Training

Reinforcement learning training for Unitree H1 humanoid robot walking, using the Genesis physics simulator and PPO.

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

## 🎬 Visualization

```bash
python play.py checkpoints/h1_walk_000060.pt
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `videos/play.mp4` | Output video path |
| `--steps` | `600` | Simulation steps to record |
| `--vx` | `1.0` | Commanded forward speed (m/s) |
| `--cpu` | off | Use CPU backend |

## 📁 Project Structure

```
├── train.py          # Training entry point
├── env.py            # H1WalkingEnv (Genesis + reward shaping)
├── ppo.py            # PPO + ActorCritic implementation
├── play.py           # Policy visualization → video
├── robot/urdf/       # H1 URDF model
├── checkpoints/      # Saved policy checkpoints
├── runs/             # TensorBoard logs
└── videos/           # Rendered videos
```
