#!/bin/bash
# V2 full pipeline: train → play final → shutdown
set -e

cd /root/autodl-tmp/h1_genesis_train
PYTHON=/root/miniconda3/envs/unitree-mujoco/bin/python

echo "[v2] Starting training at $(date)"
$PYTHON -u train.py > train_v2.log 2>&1

echo "[v2] Training complete at $(date). Running play..."
$PYTHON -u play.py checkpoints_v2/h1_walk_final.pt \
    --steps 1000 --out videos_v2/play_final.mp4 \
    --vx 1.0 --cpu >> train_v2.log 2>&1

echo "[v2] Play done at $(date). Shutting down in 60s... (Ctrl+C to cancel)"
sleep 60
shutdown -h now
