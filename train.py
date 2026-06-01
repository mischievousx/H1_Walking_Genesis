import os
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter

# Force stdout/stderr to flush immediately so logs appear in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from env import H1WalkingEnv
from ppo import PPO

# ── Hyperparameters ────────────────────────────────────────────────────────────
NUM_ENVS        = 8192
ROLLOUT_STEPS   = 24           # short sequences for LSTM BPTT (matches rsl_rl default)
N_UPDATES       = 10_000       # matches Unitree's max_iterations
LR_INIT         = 1e-3
DESIRED_KL      = 0.01   # adaptive LR target (matches rsl_rl)
SAVE_INTERVAL   = 1000         # save checkpoint every N updates
LOG_INTERVAL    = 10
SAVE_DIR        = 'checkpoints_v2'
TB_DIR          = 'runs_v2'
# ──────────────────────────────────────────────────────────────────────────────


def latest_checkpoint(save_dir):
    ckpts = [f for f in os.listdir(save_dir) if f.startswith('h1_walk_') and f.endswith('.pt')]
    if not ckpts:
        return None
    numbered = [f for f in ckpts if f not in ('h1_walk_final.pt', 'h1_walk_best.pt')]
    pool = sorted(numbered) if numbered else ckpts
    return os.path.join(save_dir, pool[-1])


def main():
    resume   = '--resume' in sys.argv
    save_dir = SAVE_DIR
    tb_dir   = TB_DIR
    if '--save-dir' in sys.argv:
        save_dir = sys.argv[sys.argv.index('--save-dir') + 1]
        tb_dir   = save_dir.replace('checkpoints', 'runs')

    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    env = H1WalkingEnv(num_envs=NUM_ENVS, device='cuda')
    ppo = PPO(
        num_obs       = env.num_obs,
        num_actions   = env.num_actions,
        device        = 'cuda',
        lr            = LR_INIT,
        rollout_steps = ROLLOUT_STEPS,
        num_envs      = NUM_ENVS,
        ent_coef      = 0.01,
    )

    start_update  = 1
    total_steps   = 0
    best_mean_rew = -float('inf')

    if resume:
        ckpt_path = latest_checkpoint(save_dir)
        if ckpt_path:
            ckpt = ppo.load(ckpt_path)
            saved_update = ckpt.get('update', 0)
            if saved_update >= N_UPDATES:
                start_update = 1
                total_steps  = 0
                print(f"Loaded weights from {ckpt_path} (restarting counters)")
            else:
                start_update = saved_update + 1
                total_steps  = ckpt.get('total_steps', 0)
                print(f"Resumed from {ckpt_path}  (update {start_update}, steps {total_steps/1e6:.1f}M)")
        else:
            print("No checkpoint found, starting from scratch.")
        best_path = os.path.join(save_dir, 'h1_walk_best.pt')
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location='cpu')
            best_mean_rew = best_ckpt.get('best_mean_rew', -float('inf'))
            print(f"Best reward so far: {best_mean_rew:.3f}")

    t0  = time.time() - total_steps / max(1, NUM_ENVS * 50)
    obs = env.reset()

    # Per-env LSTM hidden state, carried across rollouts
    hidden = ppo.ac.init_hidden(NUM_ENVS, 'cuda')

    print(f"Training H1 walking (LSTM policy) | {NUM_ENVS} envs | "
          f"{ROLLOUT_STEPS} steps/rollout | {N_UPDATES} updates total")
    print(f"TensorBoard: tensorboard --logdir {os.path.abspath(tb_dir)}")

    lr = LR_INIT
    ppo.set_lr(lr)

    for update in range(start_update, N_UPDATES + 1):
        # Snapshot hidden state at rollout start (needed for BPTT during update)
        ppo.start_rollout(hidden)

        ep_rew = 0.
        for _ in range(ROLLOUT_STEPS):
            with torch.no_grad():
                action, log_prob, value, new_hidden = ppo.ac.act(obs, hidden)

            next_obs, reward, done = env.step(action)

            # Zero hidden state for envs whose episode just ended
            done_mask = done.float().unsqueeze(0).unsqueeze(-1)   # (1, N, 1)
            new_hidden = (new_hidden[0] * (1 - done_mask),
                          new_hidden[1] * (1 - done_mask))

            ppo.store(obs, action, reward, value, log_prob, done)
            hidden      = new_hidden
            obs         = next_obs
            total_steps += NUM_ENVS
            ep_rew      += reward.mean().item()

        metrics     = ppo.update(obs, hidden)
        mean_ep_rew = ep_rew / ROLLOUT_STEPS

        # Adaptive LR: adjust based on KL divergence (matches rsl_rl)
        kl = metrics.get('kl', 0.)
        if kl > DESIRED_KL * 2.0:
            lr = max(lr / 1.5, 1e-5)
        elif kl < DESIRED_KL / 2.0:
            lr = min(lr * 1.5, 1e-2)
        ppo.set_lr(lr)

        if update % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            fps     = total_steps / elapsed
            print(
                f"[{update:5d}/{N_UPDATES}] "
                f"steps={total_steps/1e6:6.1f}M  "
                f"fps={fps:,.0f}  "
                f"lr={lr:.2e}  "
                f"rew={mean_ep_rew:+.3f}  "
                f"ploss={metrics['policy_loss']:+.4f}  "
                f"vloss={metrics['value_loss']:.4f}  "
                f"ent={metrics['entropy']:.3f}  "
                f"kl={metrics['kl']:.4f}"
            )
            writer.add_scalar('train/reward',      mean_ep_rew,           total_steps)
            writer.add_scalar('train/policy_loss', metrics['policy_loss'], total_steps)
            writer.add_scalar('train/value_loss',  metrics['value_loss'],  total_steps)
            writer.add_scalar('train/entropy',     metrics['entropy'],     total_steps)
            writer.add_scalar('train/fps',         fps,                    total_steps)
            writer.add_scalar('train/lr',          lr,                     total_steps)
            writer.add_scalar('train/kl',          metrics['kl'],          total_steps)

        if mean_ep_rew > best_mean_rew:
            best_mean_rew = mean_ep_rew
            best_path = os.path.join(save_dir, 'h1_walk_best.pt')
            ppo.save(best_path, extra=dict(update=update, total_steps=total_steps, best_mean_rew=best_mean_rew))
            writer.add_scalar('train/best_reward', best_mean_rew, total_steps)
            print(f"  → new best reward {best_mean_rew:+.3f}, saved {best_path}")

        if update % SAVE_INTERVAL == 0:
            path = os.path.join(save_dir, f'h1_walk_{update:06d}.pt')
            ppo.save(path, extra=dict(update=update, total_steps=total_steps))
            print(f"  → saved {path}")

    ppo.save(os.path.join(save_dir, 'h1_walk_final.pt'),
             extra=dict(update=N_UPDATES, total_steps=total_steps))
    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    main()
