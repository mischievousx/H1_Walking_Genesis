import os
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from env import H1WalkingEnv
from ppo import PPO

# ── Hyperparameters ────────────────────────────────────────────────────────────
NUM_ENVS        = 4096
ROLLOUT_STEPS   = 2048
TOTAL_STEPS     = 2_000_000_000  # 续训目标：共 238 次 update
LR_INIT         = 1e-3
LR_FINAL        = 1e-4
SAVE_INTERVAL   = 10            # save checkpoint every N updates
LOG_INTERVAL    = 10            # print + tensorboard every N updates
SAVE_DIR        = 'checkpoints'
TB_DIR          = 'runs'        # tensorboard log dir
# ──────────────────────────────────────────────────────────────────────────────


def latest_checkpoint(save_dir):
    """Return the most recent checkpoint (numbered > final), or None."""
    ckpts = [f for f in os.listdir(save_dir) if f.startswith('h1_walk_') and f.endswith('.pt')]
    if not ckpts:
        return None
    # Prefer numbered checkpoints; fall back to final
    numbered = [f for f in ckpts if f != 'h1_walk_final.pt']
    pool = sorted(numbered) if numbered else ckpts
    return os.path.join(save_dir, pool[-1])


def main():
    resume = '--resume' in sys.argv

    os.makedirs(SAVE_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=TB_DIR)

    env = H1WalkingEnv(num_envs=NUM_ENVS, device='cuda')
    ppo = PPO(
        num_obs       = env.num_obs,
        num_actions   = env.num_actions,
        device        = 'cuda',
        lr            = LR_INIT,
        rollout_steps = ROLLOUT_STEPS,
        num_envs      = NUM_ENVS,
    )

    n_updates    = TOTAL_STEPS // (ROLLOUT_STEPS * NUM_ENVS)
    start_update = 1
    total_steps  = 0

    if resume:
        ckpt_path = latest_checkpoint(SAVE_DIR)
        if ckpt_path:
            ckpt = ppo.load(ckpt_path)
            saved_update = ckpt.get('update', 0)
            # If checkpoint is from a completed run, use weights only and restart counters
            if saved_update >= n_updates:
                start_update = 1
                total_steps  = 0
                print(f"Loaded weights from {ckpt_path} (was update {saved_update}, restarting from update 1)")
            else:
                start_update = saved_update + 1
                total_steps  = ckpt.get('total_steps', 0)
                print(f"Resumed from {ckpt_path}  (update {start_update}, steps {total_steps/1e6:.1f}M)")
        else:
            print("No checkpoint found, starting from scratch.")

    t0 = time.time() - total_steps / max(1, NUM_ENVS * 50)  # approximate elapsed

    obs = env.reset()

    print(f"Training H1 walking | {NUM_ENVS} envs | "
          f"{ROLLOUT_STEPS} steps/rollout | target {n_updates} updates total")
    print(f"TensorBoard: tensorboard --logdir {os.path.abspath(TB_DIR)}")

    for update in range(start_update, n_updates + 1):
        # ── Cosine LR decay ───────────────────────────────────────────────────
        frac = update / n_updates
        lr   = LR_FINAL + 0.5 * (LR_INIT - LR_FINAL) * (1. + torch.cos(torch.tensor(frac * 3.14159)).item())
        ppo.set_lr(lr)

        # ── Collect rollout ───────────────────────────────────────────────────
        ep_rew = 0.
        for _ in range(ROLLOUT_STEPS):
            with torch.no_grad():
                action, log_prob, value = ppo.ac.act(obs)
            next_obs, reward, done = env.step(action)
            ppo.store(obs, action, reward, value, log_prob, done)
            obs         = next_obs
            total_steps += NUM_ENVS
            ep_rew      += reward.mean().item()

        # ── Update ───────────────────────────────────────────────────────────
        metrics     = ppo.update(obs)
        mean_ep_rew = ep_rew / ROLLOUT_STEPS

        # ── Logging ──────────────────────────────────────────────────────────
        if update % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            fps     = total_steps / elapsed

            print(
                f"[{update:5d}/{n_updates}] "
                f"steps={total_steps/1e6:6.1f}M  "
                f"fps={fps:,.0f}  "
                f"lr={lr:.2e}  "
                f"rew={mean_ep_rew:+.3f}  "
                f"ploss={metrics['policy_loss']:+.4f}  "
                f"vloss={metrics['value_loss']:.4f}  "
                f"ent={metrics['entropy']:.3f}"
            )

            writer.add_scalar('train/reward',      mean_ep_rew,           total_steps)
            writer.add_scalar('train/policy_loss', metrics['policy_loss'], total_steps)
            writer.add_scalar('train/value_loss',  metrics['value_loss'],  total_steps)
            writer.add_scalar('train/entropy',     metrics['entropy'],     total_steps)
            writer.add_scalar('train/fps',         fps,                    total_steps)
            writer.add_scalar('train/lr',          lr,                     total_steps)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if update % SAVE_INTERVAL == 0:
            path = os.path.join(SAVE_DIR, f'h1_walk_{update:06d}.pt')
            ppo.save(path, extra=dict(update=update, total_steps=total_steps))
            print(f"  → saved {path}")

    # Final save
    ppo.save(os.path.join(SAVE_DIR, 'h1_walk_final.pt'),
             extra=dict(update=n_updates, total_steps=total_steps))
    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    main()
