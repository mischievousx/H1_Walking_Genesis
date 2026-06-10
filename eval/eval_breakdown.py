"""
Reward-component breakdown evaluation.

Runs a trained policy inside H1WalkingEnv with pinned straight-line velocity
commands (vx in {0.3, 0.6, 0.9} m/s, vy=0, heading=0) and reports the mean
*raw* (unscaled) value of each reward component plus the total clipped reward,
averaged across the tested speeds.

Usage:
    python eval_breakdown.py --checkpoint checkpoints/checkpoints_v4_r48s/h1_walk_best.pt
    python eval_breakdown.py --compare
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

TRACKED = ['tracking_lin_vel', 'dof_acc', 'action_rate',
           'ang_vel_xy', 'feet_swing_height', 'contact_no_vel']

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
    """Returns per-recorded-step mean-over-envs samples for each tracked
    component plus the total reward, so the caller can pool mean/std."""
    obs, _ = env.reset()
    hidden = ac.init_hidden(env.num_envs, device)

    samples = {k: [] for k in TRACKED}
    total_samples = []

    for step_i in range(WARMUP_STEPS + RECORD_STEPS):
        env.commands[:, 0]    = vx
        env.commands[:, 1]    = 0.0
        env.heading_target[:] = 0.0

        action, hidden = ac.act_deterministic(obs, hidden)
        obs, _, rew, _ = env.step(action)

        if step_i >= WARMUP_STEPS:
            for k in TRACKED:
                samples[k].append(env.last_reward_raw[k].mean().item())
            total_samples.append(rew.mean().item())

    return samples, total_samples


def _mean_std(values):
    t = torch.tensor(values)
    return t.mean().item(), t.std().item()


def eval_checkpoint(env, checkpoint, device):
    ac = load_policy(checkpoint, device)

    pooled = {k: [] for k in TRACKED}
    pooled_total = []
    for vx in TEST_SPEEDS:
        samples, total_samples = eval_speed(env, ac, vx, device)
        m, s = _mean_std(total_samples)
        print(f"  vx={vx:.1f}  total={m:.4f}±{s:.4f}  " +
              "  ".join(f"{k}={_mean_std(samples[k])[0]:.4g}" for k in TRACKED))
        for k in TRACKED:
            pooled[k].extend(samples[k])
        pooled_total.extend(total_samples)

    comp_stats  = {k: _mean_std(pooled[k]) for k in TRACKED}
    total_stats = _mean_std(pooled_total)
    return comp_stats, total_stats


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

        print("\n\n=== Comparison (mean ± std, pooled over vx = 0.3/0.6/0.9 m/s and recorded steps) ===")
        names = list(CHECKPOINTS.keys())
        header = f"{'component':<20}" + "".join(f"{n:>26}" for n in names)
        print(header)
        for k in TRACKED:
            row = f"{k:<20}" + "".join(f"{results[n][0][k][0]:>14.6g} ± {results[n][0][k][1]:<8.4g}" for n in names)
            print(row)
        row = f"{'total_reward':<20}" + "".join(f"{results[n][1][0]:>14.6g} ± {results[n][1][1]:<8.4g}" for n in names)
        print(row)
    else:
        checkpoint = args.checkpoint or CHECKPOINTS['v4_r48s']
        print(f"\n=== {checkpoint} ===")
        comp, total = eval_checkpoint(env, checkpoint, device)
        print(f"\nPooled mean ± std (vx = 0.3/0.6/0.9 m/s):")
        for k in TRACKED:
            print(f"  {k:<20} {comp[k][0]:.6g} ± {comp[k][1]:.4g}")
        print(f"  {'total_reward':<20} {total[0]:.6g} ± {total[1]:.4g}")


if __name__ == '__main__':
    main()
