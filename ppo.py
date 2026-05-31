import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions, hidden=(512, 256, 128)):
        super().__init__()

        def mlp(in_dim, dims, out_dim, out_scale=1.0):
            layers, d = [], in_dim
            for h in dims:
                layers += [nn.Linear(d, h), nn.ELU()]
                d = h
            out = nn.Linear(d, out_dim)
            nn.init.orthogonal_(out.weight, gain=out_scale)
            nn.init.zeros_(out.bias)
            layers.append(out)
            return nn.Sequential(*layers)

        self.actor  = mlp(num_obs, hidden, num_actions, out_scale=0.01)
        self.critic = mlp(num_obs, hidden, 1,           out_scale=1.0)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.actor[-1] and m is not self.critic[-1]:
                nn.init.orthogonal_(m.weight, gain=1.414)
                nn.init.zeros_(m.bias)

    def _dist(self, obs):
        return Normal(self.actor(obs), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs):
        dist   = self._dist(obs)
        action = dist.sample()
        lp     = dist.log_prob(action).sum(-1)
        value  = self.critic(obs).squeeze(-1)
        return action, lp, value

    def evaluate(self, obs, actions):
        dist    = self._dist(obs)
        lp      = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value   = self.critic(obs).squeeze(-1)
        return lp, entropy, value

    @torch.no_grad()
    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)


class PPO:
    def __init__(self, num_obs, num_actions, device='cuda',
                 lr=1e-3, gamma=0.99, lam=0.95,
                 clip_eps=0.2, ent_coef=0.01, val_coef=1.0,
                 n_epochs=5, minibatch=16384,
                 rollout_steps=2048, num_envs=4096):

        self.device        = device
        self.gamma         = gamma
        self.lam           = lam
        self.clip_eps      = clip_eps
        self.ent_coef      = ent_coef
        self.val_coef      = val_coef
        self.n_epochs      = n_epochs
        self.minibatch     = minibatch
        self.rollout_steps = rollout_steps
        self.num_envs      = num_envs

        self.ac        = ActorCritic(num_obs, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

        T, N = rollout_steps, num_envs
        self.obs_buf  = torch.zeros(T, N, num_obs,     device=device)
        self.act_buf  = torch.zeros(T, N, num_actions, device=device)
        self.rew_buf  = torch.zeros(T, N,              device=device)
        self.val_buf  = torch.zeros(T, N,              device=device)
        self.lp_buf   = torch.zeros(T, N,              device=device)
        self.done_buf = torch.zeros(T, N,              device=device)
        self.ptr      = 0

    def store(self, obs, action, reward, value, log_prob, done):
        self.obs_buf [self.ptr] = obs
        self.act_buf [self.ptr] = action
        self.rew_buf [self.ptr] = reward
        self.val_buf [self.ptr] = value
        self.lp_buf  [self.ptr] = log_prob
        self.done_buf[self.ptr] = done.float()
        self.ptr += 1

    def update(self, last_obs):
        assert self.ptr == self.rollout_steps

        last_val = self.ac.get_value(last_obs)

        # GAE-lambda
        adv = torch.zeros_like(self.rew_buf)
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.rollout_steps)):
            nv   = last_val if t == self.rollout_steps - 1 else self.val_buf[t + 1]
            mask = 1. - self.done_buf[t]
            delta = self.rew_buf[t] + self.gamma * nv * mask - self.val_buf[t]
            gae   = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae

        ret = adv + self.val_buf

        # Flatten
        obs_f  = self.obs_buf.reshape(-1, self.obs_buf.shape[-1])
        act_f  = self.act_buf.reshape(-1, self.act_buf.shape[-1])
        lp_f   = self.lp_buf.reshape(-1)
        ret_f  = ret.reshape(-1)
        adv_f  = adv.reshape(-1)
        adv_f  = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        total  = obs_f.shape[0]
        stats  = dict(policy_loss=0., value_loss=0., entropy=0.)
        n_mb   = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(total, device=self.device)
            for start in range(0, total, self.minibatch):
                idx = perm[start : start + self.minibatch]

                new_lp, ent, new_val = self.ac.evaluate(obs_f[idx], act_f[idx])

                ratio  = (new_lp - lp_f[idx]).exp()
                mb_adv = adv_f[idx]

                pg = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps),
                ).mean()

                vl = 0.5 * (new_val - ret_f[idx]).pow(2).mean()
                el = -ent.mean()

                loss = pg + self.val_coef * vl + self.ent_coef * el

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.optimizer.step()

                stats['policy_loss'] += pg.item()
                stats['value_loss']  += vl.item()
                stats['entropy']     += ent.mean().item()
                n_mb += 1

        self.ptr = 0
        return {k: v / n_mb for k, v in stats.items()}

    def set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def save(self, path, extra=None):
        ckpt = dict(
            model=self.ac.state_dict(),
            optim=self.optimizer.state_dict(),
        )
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        return ckpt
