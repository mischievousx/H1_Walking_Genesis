"""PPO with LSTM actor-critic, faithfully ported from rsl_rl (Unitree training stack).

Key differences from previous version:
- No reward normalization (rsl_rl does not normalize rewards)
- Adaptive learning rate based on KL divergence (desired_kl=0.01)
- LR clamped to [1e-5, 1e-2]
"""
import torch
import torch.nn as nn
from torch.distributions import Normal

RNN_HIDDEN = 64


def _mlp(in_dim, hidden_dims, out_dim, out_gain=1.0):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ActorCriticRecurrent(nn.Module):
    """
    Asymmetric actor-critic with LSTM, matching H1RoughCfgPPO:
      actor_hidden_dims  = [32]
      critic_hidden_dims = [32]
      rnn_hidden_size    = 64
      init_noise_std     = 0.8
    """

    def __init__(self, num_obs, num_actions, num_privileged_obs=None, rnn_hidden=RNN_HIDDEN):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.priv_dim   = (num_privileged_obs - num_obs) if num_privileged_obs else 0

        self.memory = nn.LSTM(num_obs, rnn_hidden, num_layers=1, batch_first=False)
        self.actor  = _mlp(rnn_hidden, (32,), num_actions, out_gain=0.01)
        self.critic = _mlp(rnn_hidden + self.priv_dim, (32,), 1, out_gain=1.0)
        self.std = nn.Parameter(torch.full((num_actions,), 0.8))

        for name, p in self.memory.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def init_hidden(self, n: int, device) -> tuple:
        z = torch.zeros(1, n, self.rnn_hidden, device=device)
        return z, z.clone()

    def _std(self):
        return self.std.clamp(min=1e-4)

    def _critic_input(self, feat, priv_obs):
        if self.priv_dim > 0 and priv_obs is not None:
            return torch.cat([feat, priv_obs[:, :self.priv_dim]], dim=-1)
        return feat

    @torch.no_grad()
    def act(self, obs, hidden, priv_obs=None):
        x, new_hid = self.memory(obs.unsqueeze(0), hidden)
        x    = x.squeeze(0)
        mu   = self.actor(x)
        dist = Normal(mu, self._std())
        act  = dist.sample()
        lp   = dist.log_prob(act).sum(-1)
        val  = self.critic(self._critic_input(x, priv_obs)).squeeze(-1)
        return act, lp, val, new_hid

    @torch.no_grad()
    def act_deterministic(self, obs, hidden):
        x, new_hid = self.memory(obs.unsqueeze(0), hidden)
        return self.actor(x.squeeze(0)), new_hid

    @torch.no_grad()
    def get_value(self, obs, hidden, priv_obs=None):
        x, _ = self.memory(obs.unsqueeze(0), hidden)
        return self.critic(self._critic_input(x.squeeze(0), priv_obs)).squeeze(-1)

    def evaluate_sequence(self, obs_seq, done_seq, init_hidden, actions, priv_obs_seq=None):
        T, N, _ = obs_seq.shape
        h, c = init_hidden
        features = []
        for t in range(T):
            x, (h, c) = self.memory(obs_seq[t].unsqueeze(0), (h, c))
            features.append(x.squeeze(0))
            mask = (1.0 - done_seq[t]).unsqueeze(0).unsqueeze(-1)
            h = h * mask
            c = c * mask

        feat     = torch.stack(features).reshape(T * N, -1)
        act_flat = actions.reshape(T * N, -1)

        dist = Normal(self.actor(feat), self._std())
        lp   = dist.log_prob(act_flat).sum(-1)
        ent  = dist.entropy().sum(-1)

        if self.priv_dim > 0 and priv_obs_seq is not None:
            extra    = priv_obs_seq[:, :, :self.priv_dim].reshape(T * N, -1)
            critic_in = torch.cat([feat, extra], dim=-1)
        else:
            critic_in = feat
        val = self.critic(critic_in).squeeze(-1)

        return lp, ent, val


class PPO:
    def __init__(self, num_obs, num_actions, num_privileged_obs=None,
                 device='cuda', lr=1e-3,
                 gamma=0.99, lam=0.95,
                 clip_eps=0.2, ent_coef=0.01, val_coef=1.0,
                 n_epochs=5, num_mini_batches=4,
                 desired_kl=0.01,
                 rollout_steps=24, num_envs=4096):

        self.device           = device
        self.gamma            = gamma
        self.lam              = lam
        self.clip_eps         = clip_eps
        self.ent_coef         = ent_coef
        self.val_coef         = val_coef
        self.n_epochs         = n_epochs
        self.num_mini_batches = num_mini_batches
        self.desired_kl       = desired_kl
        self.rollout_steps    = rollout_steps
        self.num_envs         = num_envs
        self.has_priv         = num_privileged_obs is not None
        self.lr               = lr

        self.ac        = ActorCriticRecurrent(num_obs, num_actions, num_privileged_obs).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

        T, N, H = rollout_steps, num_envs, RNN_HIDDEN
        priv_dim = num_privileged_obs if num_privileged_obs else num_obs
        self.obs_buf  = torch.zeros(T, N, num_obs,     device=device)
        self.priv_buf = torch.zeros(T, N, priv_dim,    device=device)
        self.act_buf  = torch.zeros(T, N, num_actions, device=device)
        self.rew_buf  = torch.zeros(T, N,              device=device)
        self.val_buf  = torch.zeros(T, N,              device=device)
        self.lp_buf   = torch.zeros(T, N,              device=device)
        self.done_buf = torch.zeros(T, N,              device=device)

        self.init_h = torch.zeros(1, N, H, device=device)
        self.init_c = torch.zeros(1, N, H, device=device)
        self.ptr    = 0

    def start_rollout(self, hidden):
        self.init_h.copy_(hidden[0])
        self.init_c.copy_(hidden[1])

    def store(self, obs, priv_obs, action, reward, value, log_prob, done):
        self.obs_buf [self.ptr] = obs
        self.priv_buf[self.ptr] = priv_obs
        self.act_buf [self.ptr] = action
        self.rew_buf [self.ptr] = reward      # raw reward, no normalization
        self.val_buf [self.ptr] = value
        self.lp_buf  [self.ptr] = log_prob
        self.done_buf[self.ptr] = done.float()
        self.ptr += 1

    def update(self, last_obs, last_hidden, last_priv_obs=None):
        assert self.ptr == self.rollout_steps

        last_val = self.ac.get_value(last_obs, last_hidden, last_priv_obs)

        adv = torch.zeros_like(self.rew_buf)
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.rollout_steps)):
            nv    = last_val if t == self.rollout_steps - 1 else self.val_buf[t + 1]
            mask  = 1.0 - self.done_buf[t]
            delta = self.rew_buf[t] + self.gamma * nv * mask - self.val_buf[t]
            gae   = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
        ret = adv + self.val_buf

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T, N  = self.rollout_steps, self.num_envs
        mb_size = N // self.num_mini_batches

        stats = dict(policy_loss=0., value_loss=0., entropy=0., kl=0.)
        n_mb  = 0

        for _ in range(self.n_epochs):
            env_perm = torch.randperm(N, device=self.device)
            for start in range(0, N, mb_size):
                ids = env_perm[start : start + mb_size]

                obs_mb  = self.obs_buf [:, ids, :]
                priv_mb = self.priv_buf[:, ids, :]
                act_mb  = self.act_buf [:, ids, :]
                done_mb = self.done_buf[:, ids]
                lp_old  = self.lp_buf  [:, ids].reshape(-1)
                val_old = self.val_buf [:, ids].reshape(-1)
                ret_mb  = ret          [:, ids].reshape(-1)
                adv_mb  = adv          [:, ids].reshape(-1)
                init_h  = self.init_h[:, ids, :]
                init_c  = self.init_c[:, ids, :]

                lp, ent, val = self.ac.evaluate_sequence(
                    obs_mb, done_mb, (init_h, init_c), act_mb,
                    priv_mb if self.has_priv else None,
                )

                ratio = (lp - lp_old).exp()
                pg = torch.max(
                    -adv_mb * ratio,
                    -adv_mb * ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps),
                ).mean()

                val_clipped = val_old + (val - val_old).clamp(-self.clip_eps, self.clip_eps)
                vl = 0.5 * torch.max(
                    (val         - ret_mb).pow(2),
                    (val_clipped - ret_mb).pow(2),
                ).mean()

                loss = pg + self.val_coef * vl - self.ent_coef * ent.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.optimizer.step()

                with torch.no_grad():
                    kl_approx = (lp_old - lp).mean().item()
                stats['policy_loss'] += pg.item()
                stats['value_loss']  += vl.item()
                stats['entropy']     += ent.mean().item()
                stats['kl']          += kl_approx
                n_mb += 1

        self.ptr = 0
        return {k: v / n_mb for k, v in stats.items()}

    def set_lr(self, lr):
        self.lr = lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def save(self, path, extra=None):
        ckpt = dict(
            model=self.ac.state_dict(),
            optim=self.optimizer.state_dict(),
            lr=self.lr,
        )
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.ac.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        if 'lr' in ckpt:
            self.lr = ckpt['lr']
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.lr
        return ckpt
