# H1 Walking RL Training

Reinforcement learning training for Unitree H1 humanoid robot walking, using the Genesis physics simulator and PPO.

## Requirements

- NVIDIA GPU with CUDA 12.x (tested on RTX 5090 32GB)
- Conda

## Environment Setup

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

## Training

```bash
python -u train.py
```

Logs are printed every 10 updates. TensorBoard:

```bash
tensorboard --logdir runs
```

## Visualization

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

## Project Structure

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
