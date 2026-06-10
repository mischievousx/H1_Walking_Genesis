"""
Joint-smoothness comparison across the three v4_r48s ablation checkpoints.

Runs each policy inside H1WalkingEnv with pinned straight-line velocity
commands (vx in {0.3, 0.6, 0.9} m/s, vy=0, heading=0) and derives, from the
raw per-joint kinematic trajectories (env.last_dof_vel / env.last_dof_acc):

  dof_vel_jitter : for each (env, joint), the std over time of the
                   step-to-step joint-velocity difference Δv[t] = v[t]-v[t-1]
                   — high-frequency "buzz" riding on top of the gait's
                   intended velocity profile. Lower = smoother.
  dof_jerk       : for each (env, joint), the mean |Δacc[t]/dt| over time
                   — the joint's 3rd-derivative-of-position magnitude, a
                   standard motion-smoothness metric. Lower = smoother.

Both are pooled across the three speeds and reported as population
mean ± std over (env x joint x speed).

Usage:
    python eval_smoothness.py --checkpoint checkpoints/checkpoints_v4_r48s/h1_walk_best.pt
    python eval_smoothness.py --compare
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quadrants as _qd
_qd_init_orig = _qd.init
def _qd_init_patched(**kwargs):
    kwargs.setdefault('device_memory_GB', 16.0)
    return _qd_init_orig(**kwargs)
_qd.init = _qd_init_patched

import torch

from env import H1WalkingEnv
from ppo import ActorCriticRecurrent

NUM_ENVS      = 256
TEST_SPEEDS   = (0.3, 0.6, 0.9)
WARMUP_STEPS  = 100
RECORD_STEPS  = 300

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


@torch.no_grad()
def eval_speed(env, ac, vx, device):
    """Records the full per-step joint-velocity / joint-acceleration
    trajectories and returns flattened per-(env, joint) jitter / jerk
    statistics for this speed."""
    obs, _ = env.reset()
    hidden = ac.init_hidden(env.num_envs, device)

    vel_hist, acc_hist = [], []
    for step_i in range(WARMUP_STEPS + RECORD_STEPS):
        env.commands[:, 0]    = vx
        env.commands[:, 1]    = 0.0
        env.heading_target[:] = 0.0

        action, hidden = ac.act_deterministic(obs, hidden)
        obs, _, rew, _ = env.step(action)

        if step_i >= WARMUP_STEPS:
            vel_hist.append(env.last_dof_vel.clone())
            acc_hist.append(env.last_dof_acc.clone())

    vel = torch.stack(vel_hist, dim=0)             # [T, num_envs, num_joints]
    acc = torch.stack(acc_hist, dim=0)             # [T, num_envs, num_joints]

    dvel = vel[1:] - vel[:-1]                      # Δv[t] = v[t] - v[t-1]
    jerk = (acc[1:] - acc[:-1]) / env.dt           # Δacc[t]/dt  (3rd derivative of pos)

    jitter_per_unit = dvel.std(dim=0)              # [num_envs, num_joints], std over time
    jerk_per_unit   = jerk.abs().mean(dim=0)       # [num_envs, num_joints], mean |.| over time

    return jitter_per_unit.flatten(), jerk_per_unit.flatten()


def _mean_std(t):
    return t.mean().item(), t.std().item()


def eval_checkpoint(env, checkpoint, device):
    ac = load_policy(checkpoint, device)

    pooled_jitter, pooled_jerk = [], []
    for vx in TEST_SPEEDS:
        jitter, jerk = eval_speed(env, ac, vx, device)
        jm, js = _mean_std(jitter)
        km, ks = _mean_std(jerk)
        print(f"  vx={vx:.1f}  dof_vel_jitter={jm:.4g}±{js:.4g}   dof_jerk={km:.4g}±{ks:.4g}")
        pooled_jitter.append(jitter)
        pooled_jerk.append(jerk)

    pooled_jitter = torch.cat(pooled_jitter)
    pooled_jerk   = torch.cat(pooled_jerk)
    return _mean_std(pooled_jitter), _mean_std(pooled_jerk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Single checkpoint to evaluate')
    parser.add_argument('--compare', action='store_true',
                        help='Evaluate all three v4_r48s checkpoints and print a comparison table')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else 'cuda'
    env = H1WalkingEnv(num_envs=NUM_ENVS, device=device)

    if args.compare:
        results = {}
        for name, ckpt in CHECKPOINTS.items():
            print(f"\n=== {name} ===")
            results[name] = eval_checkpoint(env, ckpt, device)

        print("\n\n=== Comparison (mean ± std, pooled over vx = 0.3/0.6/0.9 m/s, env x joint) ===")
        names = list(CHECKPOINTS.keys())
        header = f"{'metric':<20}" + "".join(f"{n:>26}" for n in names)
        print(header)
        row = f"{'dof_vel_jitter':<20}" + "".join(f"{results[n][0][0]:>14.6g} ± {results[n][0][1]:<8.4g}" for n in names)
        print(row)
        row = f"{'dof_jerk':<20}" + "".join(f"{results[n][1][0]:>14.6g} ± {results[n][1][1]:<8.4g}" for n in names)
        print(row)
    else:
        checkpoint = args.checkpoint or CHECKPOINTS['v4_r48s']
        print(f"\n=== {checkpoint} ===")
        jitter, jerk = eval_checkpoint(env, checkpoint, device)
        print(f"\nPooled mean ± std (vx = 0.3/0.6/0.9 m/s):")
        print(f"  dof_vel_jitter      {jitter[0]:.6g} ± {jitter[1]:.4g}")
        print(f"  dof_jerk            {jerk[0]:.6g} ± {jerk[1]:.4g}")


if __name__ == '__main__':
    main()
