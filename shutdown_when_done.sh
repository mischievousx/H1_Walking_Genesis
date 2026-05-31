#!/bin/bash
# Monitors training log and shuts down the server when training completes.
LOG=/root/autodl-tmp/h1_genesis_train/train.log

echo "[monitor] Waiting for training to complete..."
echo "[monitor] Watching: $LOG"

while true; do
    if grep -q "Training complete." "$LOG" 2>/dev/null; then
        echo "[monitor] Training complete detected at $(date)."
        echo "[monitor] Final checkpoint:"
        ls -lh /root/autodl-tmp/h1_genesis_train/checkpoints/
        echo "[monitor] Shutting down in 30 seconds... (Ctrl+C to cancel)"
        sleep 30
        shutdown -h now
        exit 0
    fi
    sleep 60
done
