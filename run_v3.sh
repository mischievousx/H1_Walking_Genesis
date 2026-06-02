#!/bin/bash
# V3: Unitree-aligned config (sigma=0.25, obs noise, push robots, collision penalty)
set -e

cd /root/autodl-tmp/h1_genesis_train
PYTHON=/root/miniconda3/envs/unitree-mujoco/bin/python

echo "[v3] Starting training at $(date)"
$PYTHON -u train.py > train_v3.log 2>&1

echo "[v3] Training complete at $(date). Running play..."
$PYTHON -u play.py checkpoints_v3/h1_walk_best.pt \
    --steps 1000 --out videos_v3/play_best.mp4 \
    --vx 1.0 >> train_v3.log 2>&1

echo "[v3] Done at $(date). Shutting down in 60s... (Ctrl+C to cancel)"
sleep 60
shutdown -h now
