#!/bin/bash
echo "Waiting for training (PID 15718) to finish..."
while kill -0 15718 2>/dev/null; do
    sleep 30
done
echo "Training done. Running play..."
cd /root/autodl-tmp/h1_genesis_train
/root/miniconda3/envs/unitree-mujoco/bin/python -u play.py checkpoints/h1_walk_final.pt --out videos/play_v2_final.mp4 --steps 600
echo "Play done. Video saved to videos/play_v2_final.mp4"
