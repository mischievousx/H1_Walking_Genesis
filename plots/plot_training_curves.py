"""
Plot the training-reward and value-loss curves of the three v4_r48s ablation
runs (r48s / acc5x / acc10x) directly from their training logs
(train_v4_r48s*.log), which already contain one summary line every 10 PPO
updates:

    [  100/10000] steps=  39.3M  fps=...  lr=...  rew=+0.014  ploss=...  vloss=0.0069  ent=...  kl=...

Two panels are produced for each model on shared axes:
  - mean episodic reward  vs. update step
  - value-function loss   vs. update step  (shown on a log scale — vloss spans
    several orders of magnitude over the course of training)

Usage:
    python plot_training_curves.py
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOGS = {
    'v4_r48s':        'logs/train_v4_r48s.log',
    'v4_r48s_acc5x':  'logs/train_v4_r48s_acc5x.log',
    'v4_r48s_acc10x': 'logs/train_v4_r48s_acc10x.log',
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, 'report', 'figs', 'training_curves.png')

LINE_RE = re.compile(
    r'\[\s*(\d+)/\d+\]\s+steps=\s*([\d.]+)M\s+fps=\S+\s+lr=\S+\s+'
    r'rew=([+-][\d.]+)\s+ploss=\S+\s+vloss=([\d.]+)'
)


def _smooth(x, window=21):
    """Centered moving average; edges use a shrinking window so the
    smoothed curve still spans the full range without phase shift."""
    out = np.empty_like(x, dtype=float)
    half = window // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        out[i] = x[lo:hi].mean()
    return out


def parse_log(path):
    updates, rewards, vlosses = [], [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                updates.append(int(m.group(1)))
                rewards.append(float(m.group(3)))
                vlosses.append(float(m.group(4)))
    return np.array(updates), np.array(rewards), np.array(vlosses)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    cmap = plt.get_cmap('viridis')
    colors = [cmap(x) for x in np.linspace(0.15, 0.85, len(LOGS))]

    for (name, log_path), color in zip(LOGS.items(), colors):
        updates, rewards, vlosses = parse_log(os.path.join(ROOT, log_path))
        print(f"{name}: {len(updates)} points, "
              f"final rew={rewards[-1]:+.3f}, final vloss={vlosses[-1]:.4f}")
        axes[0].plot(updates, rewards, color=color, lw=0.7, alpha=0.25)
        axes[0].plot(updates, _smooth(rewards), color=color, lw=1.8, label=name)
        axes[1].plot(updates, vlosses, color=color, lw=0.7, alpha=0.25)
        axes[1].plot(updates, _smooth(vlosses), color=color, lw=1.8, label=name)

    axes[0].set_ylabel('mean episodic reward')
    axes[0].set_title('Training convergence: reward & value loss vs. PPO update (10000 updates total)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel('value loss (log scale)')
    axes[1].set_xlabel('PPO update step')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
