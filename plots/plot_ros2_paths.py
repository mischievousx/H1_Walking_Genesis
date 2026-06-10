"""
Plot the 2D top-down walking paths recorded by the ROS2 closed-loop route test
(play_ros2_path.py) for the three v4_r48s ablation checkpoints, overlaid on
the same axes for direct path-tracking comparison.

Each `*_path.npy` is an [N, 2] array of (x, y) base positions, one row per
0.02 s control step, recorded while the robot follows the same deterministic
6-segment route (forward → left turn → forward → right turn → forward → stop,
1250 steps total — see FIXED_ROUTE in play_ros2_path.py). Vertical markers at
the route's segment-boundary step indices are projected onto each path so the
"forward / turn / forward / turn / forward / stop" structure is visible on
the trajectory itself.

Usage:
    python plot_ros2_paths.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATHS = {
    'v4_r48s':        'videos/ros2_path/r48s_path.npy',
    'v4_r48s_acc5x':  'videos/ros2_path/acc5x_path.npy',
    'v4_r48s_acc10x': 'videos/ros2_path/acc10x_path.npy',
}

# (steps, label) — mirrors FIXED_ROUTE in play_ros2_path.py; cumulative sums
# give the step index at each segment boundary.
ROUTE_SEGMENTS = [
    (200, 'Forward'),
    (300, 'Left turn'),
    (200, 'Forward'),
    (300, 'Right turn'),
    (150, 'Forward'),
    (100, 'Stop'),
]
BOUNDARIES = np.cumsum([s for s, _ in ROUTE_SEGMENTS])  # [200,500,700,1000,1150,1250]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, 'report', 'figs', 'ros2_paths.png')


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))

    cmap = plt.get_cmap('viridis')
    colors = [cmap(x) for x in np.linspace(0.15, 0.85, len(PATHS))]

    for (name, path_file), color in zip(PATHS.items(), colors):
        p = np.load(os.path.join(ROOT, path_file))
        ax.plot(p[:, 0], p[:, 1], color=color, lw=2.0, label=name)
        ax.scatter(p[0, 0], p[0, 1], color=color, marker='o', s=60, zorder=5)
        ax.scatter(p[-1, 0], p[-1, 1], color=color, marker='X', s=90, zorder=5)
        # mark segment boundaries (turn points) along this trajectory
        for b in BOUNDARIES[:-1]:
            if b < len(p):
                ax.scatter(p[b, 0], p[b, 1], color=color, marker='|', s=140,
                           lw=2.0, zorder=4)
        print(f'{name}: start=({p[0,0]:+.2f},{p[0,1]:+.2f})  '
              f'end=({p[-1,0]:+.2f},{p[-1,1]:+.2f})  steps={len(p)}')

    ax.scatter([], [], color='gray', marker='o', s=60, label='start')
    ax.scatter([], [], color='gray', marker='X', s=90, label='end')
    ax.scatter([], [], color='gray', marker='|', s=140, lw=2.0, label='segment boundary\n(turn command issued)')

    ax.set_xlabel('world $x$ (m)')
    ax.set_ylabel('world $y$ (m)')
    ax.set_title('ROS2 closed-loop route test: top-down walking path\n'
                 '(forward $\\to$ left turn $\\to$ forward $\\to$ right turn $\\to$ forward $\\to$ stop)',
                 fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f'Saved: {OUT_PATH}')


if __name__ == '__main__':
    main()
