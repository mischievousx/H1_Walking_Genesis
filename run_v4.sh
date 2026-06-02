#!/bin/bash
# V4: full Unitree alignment + asymmetric actor-critic (privileged obs)
set -e

cd /root/autodl-tmp/h1_genesis_train
PYTHON=/root/miniconda3/envs/unitree-mujoco/bin/python

echo "[v4] Starting training at $(date)"
$PYTHON -u train.py > train_v4.log 2>&1

echo "[v4] Training complete at $(date). Running play..."
$PYTHON -u play.py checkpoints_v4/h1_walk_best.pt \
    --steps 1000 --out videos_v4/play_best.mp4 \
    --vx 1.0 >> train_v4.log 2>&1

echo "[v4] Done at $(date). Shutting down in 60s... (Ctrl+C to cancel)"
sleep 60
shutdown -h now
