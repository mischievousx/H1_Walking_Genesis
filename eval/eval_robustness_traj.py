"""
Visualize the robot's true center-of-mass (COM) trajectory under lateral push
disturbances, for each of the three v4_r48s ablation checkpoints.

Unlike eval_robustness.py (which aggregates a single survival-rate statistic
over many runs), this script runs each (checkpoint, push level) combination
ONCE — one batch of NUM_ENVS parallel environments — and directly plots the
resulting COM trajectories:

  - height z(t)        : absolute world-frame height, ground = 0
  - lateral disp. y(t) : world-frame y position relative to the push-onset
                         COM position (no natural "ground" reference for the
                         lateral axis, so we zero it at the moment of impact)

The plotted window starts at t = 0 = the moment the push impulse is applied
(the warm-up that precedes it is run, but not recorded/plotted — single-
trajectory plots turned out to be dominated by which gait phase the impulse
happens to land on, a coin-flip that has nothing to do with the policy's
actual robustness; the 256-way batch is what makes a single run trustworthy).

The true COM (rather than the pelvis/base link used elsewhere as a height
proxy) is computed as the mass-weighted average of all link positions:

    COM = sum_i(m_i * pos_i) / sum_i(m_i)

using env.robot.get_links_pos() and env.robot.get_links_inertial_mass().

As in eval_robustness.py, _check_termination is monkey-patched to a no-op for
the observation window so every env keeps producing a real trajectory for the
full window regardless of what would normally trigger an auto-reset.

One figure per checkpoint is produced, comparing all PUSH_LEVELS on the same
axes (batch mean, bold, plus a handful of individual sampled trajectories,
thin/translucent, to show spread without needing repeated runs).

Usage:
    python eval_robustness_traj.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quadrants as _qd
_qd_init_orig = _qd.init
def _qd_init_patched(**kwargs):
    kwargs.setdefault('device_memory_GB', 16.0)
    return _qd_init_orig(**kwargs)
_qd.init = _qd_init_patched

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env import H1WalkingEnv, MAX_PUSH_VEL
from ppo import ActorCriticRecurrent

NUM_ENVS         = 256     # one batch run per (checkpoint, push level); the 256 parallel
                           # samples are what let a single run show both the typical
                           # trajectory (batch mean) and its spread (sampled individuals)
                           # without needing repeated sampling
VX_CMD           = 0.5
WARMUP_STEPS     = 100     # ~2s, settle into steady walking before the push (run but not plotted —
                           # the plotted window starts at t=0 = the moment of impact)
PUSH_LEVELS      = (0.25, 0.5, 0.5 * MAX_PUSH_VEL, MAX_PUSH_VEL)   # (0.25, 0.5, 0.75, 1.5) m/s
RECOVERY_WINDOW  = 150     # ~3s post-push tracking window — this is what gets plotted, t=0 at push
N_SAMPLE_TRAJ    = 12      # number of individual trajectories overlaid (thin lines) per level

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'report', 'figs')

CHECKPOINTS = {
    'v4_r48s':        'checkpoints/checkpoints_v4_r48s/h1_walk_best.pt',
    'v4_r48s_acc5x':  'checkpoints/checkpoints_v4_r48s_acc5x/h1_walk_best.pt',
    'v4_r48s_acc10x': 'checkpoints/checkpoints_v4_r48s_acc10x/h1_walk_best.pt',
}


def load_policy(checkpoint, device):
    ac = ActorCriticRecurrent(H1WalkingEnv.num_obs, H1WalkingEnv.num_actions,
                              num_privileged_obs=H1WalkingEnv.num_privileged_obs).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    ac.load_state_dict(ckpt['model'])
    ac.eval()
    print(f"Loaded: {checkpoint}  (step {ckpt.get('total_steps', '?')})")
    return ac


def _pin_commands(env):
    env.commands[:, 0]    = VX_CMD
    env.commands[:, 1]    = 0.0
    env.heading_target[:] = 0.0


def _no_termination(env):
    """_check_termination replacement that never fires, so env.step() never
    auto-resets a env mid-window — letting us record the full, untruncated
    COM trajectory regardless of what would normally terminate it."""
    zeros = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return lambda contact_forces: zeros


def _com_pos(env, link_weights):
    """Mass-weighted center of mass, shape [num_envs, 3]."""
    links_pos = env.robot.get_links_pos()              # [num_envs, n_links, 3]
    return (links_pos * link_weights).sum(dim=1)       # [num_envs, 3]


@torch.no_grad()
def run_traj_test(env, ac, device, push_vel, link_weights):
    obs, _ = env.reset()
    hidden = ac.init_hidden(env.num_envs, device)

    # --- warm-up (not recorded): let the policy settle into steady-state
    #     walking before the push is applied; the plotted window starts at
    #     t=0 = the moment of impact, not at the start of the simulation ---
    for _ in range(WARMUP_STEPS):
        _pin_commands(env)
        action, hidden = ac.act_deterministic(obs, hidden)
        obs, _, rew, done = env.step(action)

    # COM lateral position at the instant of impact — zero point for y(t)
    y0 = _com_pos(env, link_weights)[:, 1].clone()

    # --- apply an identical lateral velocity impulse to every env ---
    root_lin_dofs = torch.arange(0, 3, device=device)
    cur  = env.robot.get_dofs_velocity(root_lin_dofs)
    push = torch.zeros(env.num_envs, 3, device=device)
    push[:, 1] = push_vel
    env.robot.set_dofs_velocity(cur + push, root_lin_dofs)

    # --- observation window: disable auto-reset-on-fall so every env keeps
    #     producing a real COM trajectory for the full window ---
    orig_check_termination = env._check_termination
    env._check_termination = _no_termination(env)
    try:
        z_hist, y_hist = [], []
        for _ in range(RECOVERY_WINDOW):
            _pin_commands(env)
            action, hidden = ac.act_deterministic(obs, hidden)
            obs, _, rew, _ = env.step(action)
            com = _com_pos(env, link_weights)
            z_hist.append(com[:, 2])
            y_hist.append(com[:, 1])
    finally:
        env._check_termination = orig_check_termination

    com_z = torch.stack(z_hist, dim=0)                       # [T, num_envs]
    com_y = torch.stack(y_hist, dim=0) - y0.unsqueeze(0)     # [T, num_envs], zeroed at impact
    return com_z.cpu().numpy(), com_y.cpu().numpy()


def plot_model(name, results, dt, out_dir):
    """results: ordered dict push_level -> (com_z [T,N], com_y [T,N])"""
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    T = next(iter(results.values()))[0].shape[0]
    t = np.arange(T) * dt   # t=0 at the moment of impact (start of the recorded window)

    cmap = plt.get_cmap('viridis')
    colors = [cmap(x) for x in np.linspace(0.15, 0.9, len(PUSH_LEVELS))]
    rng = np.random.default_rng(0)

    for (lvl, (com_z, com_y)), color in zip(results.items(), colors):
        n = com_z.shape[1]
        idx = rng.choice(n, size=min(N_SAMPLE_TRAJ, n), replace=False)
        label = f"{lvl:.2f} m/s ({lvl / MAX_PUSH_VEL:.0%} of max)"
        for i in idx:
            axes[0].plot(t, com_z[:, i], color=color, alpha=0.15, lw=0.8)
            axes[1].plot(t, com_y[:, i], color=color, alpha=0.15, lw=0.8)
        axes[0].plot(t, com_z.mean(axis=1), color=color, lw=2.2, label=label)
        axes[1].plot(t, com_y.mean(axis=1), color=color, lw=2.2, label=label)

    axes[0].axhline(0.0, color='k', lw=1.0, ls='--', alpha=0.6, label='ground (z = 0)')
    axes[0].axvline(0.0, color='r', lw=1.0, ls=':', alpha=0.7, label='impact (t = 0)')
    axes[0].set_ylabel('COM height $z$ (m, ground = 0)')
    axes[0].set_title(f'{name}: COM trajectory under lateral push impulses (vx_cmd = {VX_CMD} m/s)')
    axes[0].legend(fontsize=8, title='push magnitude', loc='lower right')
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.0, color='k', lw=1.0, ls='--', alpha=0.6)
    axes[1].axvline(0.0, color='r', lw=1.0, ls=':', alpha=0.7, label='impact (t = 0)')
    axes[1].set_ylabel(r'lateral disp. $\Delta y$ (m, impact moment = 0)')
    axes[1].set_xlabel('time (s)  [t = 0 at the moment of impact]')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f'com_traj_{name}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda'
    env = H1WalkingEnv(num_envs=NUM_ENVS, device=device)

    masses = env.robot.get_links_inertial_mass()              # [n_links]
    link_weights = (masses / masses.sum()).view(1, -1, 1)     # [1, n_links, 1]
    print(f"Robot total mass: {masses.sum().item():.2f} kg over {masses.shape[0]} links")
    print(f"vx_cmd={VX_CMD} m/s, warmup={WARMUP_STEPS} steps, "
          f"window={RECOVERY_WINDOW} steps, dt={env.dt:.3f}s, MAX_PUSH_VEL={MAX_PUSH_VEL} m/s")

    for name, ckpt in CHECKPOINTS.items():
        print(f"\n=== {name} ===")
        ac = load_policy(ckpt, device)
        results = {}
        for lvl in PUSH_LEVELS:
            print(f"  push = {lvl:.2f} m/s ...")
            results[lvl] = run_traj_test(env, ac, device, lvl, link_weights)
        plot_model(name, results, env.dt, OUT_DIR)


if __name__ == '__main__':
    main()
