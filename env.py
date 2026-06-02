import os
import torch
import genesis as gs

URDF_PATH = os.path.join(os.path.dirname(__file__), 'robot/urdf/h1.urdf')

_DEFAULT_DOF_POS = torch.tensor([
    0.,  0.,    # hip yaw  L, R
    0.,  0.,    # hip roll L, R
   -0.1,-0.1,  # hip pitch L, R
    0.3, 0.3,  # knee L, R
   -0.2,-0.2,  # ankle L, R
])
_KP = torch.tensor([150,150,150,150,150,150,200,200,40,40], dtype=torch.float32)
_KD = torch.tensor([  2,  2,  2,  2,  2,  2,  4,  4, 2, 2], dtype=torch.float32)

# DOF limits from URDF (Genesis order: L/R interleaved per joint type)
_DOF_LOWER = torch.tensor([-0.43,-0.43, -0.43,-0.43, -1.57,-1.57, -0.26,-0.26, -0.87,-0.87])
_DOF_UPPER = torch.tensor([ 0.43, 0.43,  0.43, 0.43,  1.57, 1.57,  2.05, 2.05,  0.52, 0.52])
_SOFT_LIMIT = 0.9   # fraction of hard limit before penalty kicks in

# Observation noise scales (uniform ±, applied after obs scaling)
# noise_scale * noise_level * obs_scale  (noise_level=1.0)
_NOISE_VEC = torch.tensor([
    0.05, 0.05, 0.05,               # ang_vel   (0.2 * 0.25)
    0.05, 0.05, 0.05,               # proj_g    (0.05)
    0.00, 0.00, 0.00,               # commands  (no noise)
    *([0.01] * 10),                 # dof_pos   (0.01 * 1.0)
    *([0.075] * 10),                # dof_vel   (1.5 * 0.05)
    *([0.00] * 10),                 # last_action
    0.00, 0.00,                     # sin/cos phase
])

RESAMPLE_INTERVAL = 500   # resample commands every 10 s
PUSH_INTERVAL     = 250   # push robots every 5 s (250 steps × 0.02 s)
MAX_PUSH_VEL      = 1.5   # m/s


class H1WalkingEnv:
    num_obs            = 41
    num_privileged_obs = 44   # actor obs + lin_vel(3) for critic
    num_actions        = 10

    def __init__(self, num_envs=4096, sim_dt=0.005, control_decimation=4, device='cuda'):
        self.num_envs           = num_envs
        self.device             = device
        self.sim_dt             = sim_dt
        self.control_decimation = control_decimation
        self.dt                 = sim_dt * control_decimation   # 0.02 s = 50 Hz
        self.max_episode_length = 1000

        self.default_dof_pos = _DEFAULT_DOF_POS.to(device)
        self.motor_dofs      = torch.arange(6, 16, device=device)

        self.dof_lower_soft  = (_DOF_LOWER * _SOFT_LIMIT).to(device)
        self.dof_upper_soft  = (_DOF_UPPER * _SOFT_LIMIT).to(device)
        self.noise_vec       = _NOISE_VEC.to(device)

        self.commands          = torch.zeros(num_envs, 3, device=device)
        self.heading_target    = torch.zeros(num_envs, device=device)
        self.episode_len_buf   = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.last_action       = torch.zeros(num_envs, self.num_actions, device=device)
        self.last_dof_vel      = torch.zeros(num_envs, self.num_actions, device=device)
        self.last_dof_acc      = torch.zeros(num_envs, self.num_actions, device=device)
        self.foot_air_time     = torch.zeros(num_envs, 2, device=device)
        self.last_foot_contact = torch.ones(num_envs, 2, dtype=torch.bool, device=device)
        self.phase             = torch.zeros(num_envs, device=device)
        self.push_counter      = 0

        self._build_scene()

    # ------------------------------------------------------------------
    def _build_scene(self):
        gs.init(backend=gs.cuda, logging_level='error')
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=1),
            show_viewer=False,
            show_FPS=False,
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=URDF_PATH, pos=(0., 0., 1.05))
        )
        self.scene.build(n_envs=self.num_envs, env_spacing=(2., 2.))

        self.robot.set_dofs_kp(_KP.to(self.device), self.motor_dofs)
        self.robot.set_dofs_kv(_KD.to(self.device), self.motor_dofs)

        self.foot_link_idx = [
            self.robot.get_link('left_ankle_link').idx_local,
            self.robot.get_link('right_ankle_link').idx_local,
        ]

        # Links penalized on contact (hip + knee)
        _penalize_names = [
            'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link',
            'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link',
            'left_knee_link', 'right_knee_link',
        ]
        self.penalized_link_idx = [
            self.robot.get_link(n).idx_local for n in _penalize_names
        ]

        # Link that terminates episode on contact (pelvis)
        self.pelvis_link_idx = self.robot.get_link('pelvis').idx_local
        self.pelvis_local_idx = torch.tensor([self.pelvis_link_idx], device=self.device)

    # ------------------------------------------------------------------
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)

        pos  = torch.zeros(n, 3, device=self.device); pos[:, 2] = 1.05
        quat = torch.zeros(n, 4, device=self.device); quat[:, 0] = 1.0

        self.robot.set_pos(pos, envs_idx=env_ids)
        self.robot.set_quat(quat, envs_idx=env_ids)
        self.robot.set_dofs_position(
            self.default_dof_pos.unsqueeze(0).expand(n, -1),
            self.motor_dofs, envs_idx=env_ids,
        )
        self.robot.zero_all_dofs_velocity(envs_idx=env_ids)

        # Domain randomization: friction [0.1, 1.25] and base mass shift [-1, 3] kg
        friction = torch.rand(n, device=self.device) * 1.15 + 0.1
        self.plane.set_friction_ratio(friction, envs_idx=env_ids)
        mass_shift = torch.rand(n, device=self.device) * 4.0 - 1.0
        self.robot.set_mass_shift(
            mass_shift.unsqueeze(1), self.pelvis_local_idx, envs_idx=env_ids
        )

        self.episode_len_buf[env_ids]   = 0
        self.last_action[env_ids]       = 0
        self.last_dof_vel[env_ids]      = 0
        self.last_dof_acc[env_ids]      = 0
        self.foot_air_time[env_ids]     = 0
        self.last_foot_contact[env_ids] = True
        self.phase[env_ids]             = torch.rand(n, device=self.device)
        self._sample_commands(env_ids)

        obs = self._get_obs()
        return obs, self._get_privileged_obs(obs)

    # ------------------------------------------------------------------
    def _sample_commands(self, env_ids):
        n = len(env_ids)
        vx = torch.rand(n, device=self.device) * 2.0 - 1.0   # [-1.0, 1.0] m/s
        vy = torch.rand(n, device=self.device) * 2.0 - 1.0   # [-1.0, 1.0] m/s
        self.commands[env_ids, 0] = vx
        self.commands[env_ids, 1] = vy
        self.heading_target[env_ids] = (
            torch.rand(n, device=self.device) * 2 * 3.14159 - 3.14159
        )
        small = torch.norm(self.commands[env_ids, :2], dim=1) < 0.2
        self.commands[env_ids[small], :2] = 0.
        self.heading_target[env_ids[small]] = 0.

    def _update_heading_command(self):
        quat = self.robot.get_quat()
        qw, qx, qy, qz = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        heading = torch.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        err = self.heading_target - heading
        err = ((err + 3.14159) % (2 * 3.14159)) - 3.14159
        self.commands[:, 2] = torch.clamp(0.5 * err, -1., 1.)
        no_vel = torch.norm(self.commands[:, :2], dim=1) < 0.1
        self.commands[no_vel, 2] = 0.

    def _push_robots(self):
        """Apply random velocity impulse via root joint DOFs (domain randomization)."""
        root_lin_dofs = torch.arange(0, 3, device=self.device)
        cur = self.robot.get_dofs_velocity(root_lin_dofs)
        push = (torch.rand(self.num_envs, 3, device=self.device) * 2 - 1) * MAX_PUSH_VEL
        push[:, 2] = 0.
        self.robot.set_dofs_velocity(cur + push, root_lin_dofs)

    # ------------------------------------------------------------------
    def step(self, actions):
        actions = torch.clamp(actions, -1., 1.)
        target  = self.default_dof_pos + actions * 0.25

        dof_vel_prev = self.robot.get_dofs_velocity(self.motor_dofs).clone()

        for _ in range(self.control_decimation):
            self.robot.control_dofs_position(target, self.motor_dofs)
            self.scene.step()

        self.episode_len_buf += 1
        self.phase = (self.phase + self.dt / 0.8) % 1.0

        self.push_counter += 1
        if self.push_counter % PUSH_INTERVAL == 0:
            self._push_robots()

        resample_ids = (self.episode_len_buf % RESAMPLE_INTERVAL == 0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._sample_commands(resample_ids)

        self._update_heading_command()

        contact_forces = self.robot.get_links_net_contact_force()
        foot_contact = torch.stack([
            torch.norm(contact_forces[:, self.foot_link_idx[0], :], dim=-1) > 1.0,
            torch.norm(contact_forces[:, self.foot_link_idx[1], :], dim=-1) > 1.0,
        ], dim=1)

        self.foot_air_time += self.dt

        dof_vel_now = self.robot.get_dofs_velocity(self.motor_dofs)
        dof_acc     = (dof_vel_now - dof_vel_prev) / self.dt
        dof_jerk    = (dof_acc - self.last_dof_acc) / self.dt

        rew  = self._compute_rewards(actions, dof_acc, dof_jerk, foot_contact, contact_forces)
        rew  = torch.clamp(rew, min=0.)
        done = self._check_termination(contact_forces)

        self.foot_air_time[foot_contact] = 0.
        self.last_action       = actions.clone()
        self.last_dof_vel      = dof_vel_now.clone()
        self.last_foot_contact = foot_contact.clone()
        self.last_dof_acc      = dof_acc.clone()

        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self.reset(done_ids)

        obs = self._get_obs()
        return obs, self._get_privileged_obs(obs), rew, done

    # ------------------------------------------------------------------
    def _get_obs(self):
        quat    = self.robot.get_quat()
        ang_vel = self._q_inv_rotate(quat, self.robot.get_ang())
        proj_g  = self._q_inv_rotate(
            quat,
            torch.tensor([[0., 0., -1.]], device=self.device).expand(self.num_envs, -1),
        )
        dof_pos   = self.robot.get_dofs_position(self.motor_dofs) - self.default_dof_pos
        dof_vel   = self.robot.get_dofs_velocity(self.motor_dofs)
        sin_phase = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)
        cmd_scale = torch.tensor([2., 2., 0.25], device=self.device)

        obs = torch.cat([
            ang_vel * 0.25,
            proj_g,
            self.commands * cmd_scale,
            dof_pos,
            dof_vel * 0.05,
            self.last_action,
            sin_phase,
            cos_phase,
        ], dim=-1)

        obs += (2 * torch.rand_like(obs) - 1) * self.noise_vec
        return torch.clamp(obs, -5., 5.)

    def _get_privileged_obs(self, obs):
        """44-dim privileged obs for critic: lin_vel(3) prepended to obs(41)."""
        quat    = self.robot.get_quat()
        lin_vel = self._q_inv_rotate(quat, self.robot.get_vel()) * 2.0  # lin_vel_scale=2.0
        return torch.cat([lin_vel, obs], dim=-1)

    def _q_inv_rotate(self, q, v):
        qw = q[:, 0]; qv = q[:, 1:]
        a  = v * (2. * qw**2 - 1.).unsqueeze(-1)
        b  = torch.cross(qv, v, dim=1) * (2. * qw).unsqueeze(-1)
        c  = qv * torch.sum(qv * v, dim=1, keepdim=True) * 2.
        return a - b + c

    # ------------------------------------------------------------------
    def _compute_rewards(self, actions, dof_acc, dof_jerk, foot_contact, contact_forces):
        quat    = self.robot.get_quat()
        lin_vel = self._q_inv_rotate(quat, self.robot.get_vel())
        ang_vel = self._q_inv_rotate(quat, self.robot.get_ang())
        proj_g  = self._q_inv_rotate(
            quat,
            torch.tensor([[0., 0., -1.]], device=self.device).expand(self.num_envs, -1),
        )

        # ── Velocity tracking (sigma=0.25, matches Unitree base config) ──────
        lin_err = torch.sum((self.commands[:, :2] - lin_vel[:, :2])**2, dim=1)
        ang_err = (self.commands[:, 2] - ang_vel[:, 2])**2
        r_lin   = torch.exp(-lin_err / 0.25)
        r_ang   = torch.exp(-ang_err / 0.25)

        # ── Body motion penalties ──────────────────────────────────────────
        r_lin_vel_z  = -lin_vel[:, 2]**2
        r_ang_vel_xy = -torch.sum(ang_vel[:, :2]**2, dim=1)
        r_orientation = -torch.sum(proj_g[:, :2]**2, dim=1)
        r_height      = -(self.robot.get_pos()[:, 2] - 1.05)**2

        # ── Joint penalties ────────────────────────────────────────────────
        dof_pos   = self.robot.get_dofs_position(self.motor_dofs)
        r_hip_pos = -torch.sum(dof_pos[:, [0, 1, 2, 3]]**2, dim=1)

        # ── DOF position soft limits (linear violation, matches Unitree) ──────
        upper_viol = torch.clamp(dof_pos - self.dof_upper_soft, min=0.)
        lower_viol = torch.clamp(self.dof_lower_soft - dof_pos, min=0.)
        r_dof_pos_limits = -torch.sum(upper_viol + lower_viol, dim=1)

        # ── Hip/knee collision penalty ─────────────────────────────────────
        penalized_forces = torch.norm(
            contact_forces[:, self.penalized_link_idx, :], dim=-1
        )
        r_collision = -torch.sum((penalized_forces > 0.1).float(), dim=1)

        # ── Gait phase contact timing ──────────────────────────────────────
        phase_left  = self.phase
        phase_right = (self.phase + 0.5) % 1.0
        is_stance   = torch.stack([phase_left < 0.55, phase_right < 0.55], dim=1)
        r_contact   = torch.sum((foot_contact == is_stance).float(), dim=1)

        # ── Feet swing height ──────────────────────────────────────────────
        foot_pos = self.robot.get_links_pos()
        fl_z = foot_pos[:, self.foot_link_idx[0], 2]
        fr_z = foot_pos[:, self.foot_link_idx[1], 2]
        swing = ~foot_contact
        r_swing_height = -(swing[:, 0].float() * (fl_z - 0.08)**2 +
                           swing[:, 1].float() * (fr_z - 0.08)**2)

        # ── Contact no velocity ────────────────────────────────────────────
        foot_vel = self.robot.get_links_vel()
        fl_vel   = foot_vel[:, self.foot_link_idx[0], :]
        fr_vel   = foot_vel[:, self.foot_link_idx[1], :]
        r_contact_no_vel = -(foot_contact[:, 0].float() * torch.sum(fl_vel**2, dim=1) +
                              foot_contact[:, 1].float() * torch.sum(fr_vel**2, dim=1))

        # ── Alive ─────────────────────────────────────────────────────────
        r_alive = torch.ones(self.num_envs, device=self.device)

        # ── Smoothness ────────────────────────────────────────────────────
        r_dof_acc  = -torch.sum(dof_acc**2,  dim=1) * 2.5e-7
        r_act_rate = -torch.sum((actions - self.last_action)**2, dim=1) * 0.01

        # Weights exactly matching Unitree H1RoughCfg
        return (
            1.0  * r_lin             +   # tracking_lin_vel
            0.5  * r_ang             +   # tracking_ang_vel
            2.0  * r_lin_vel_z       +   # lin_vel_z
            0.05 * r_ang_vel_xy      +   # ang_vel_xy
            1.0  * r_orientation     +   # orientation
            10.0 * r_height          +   # base_height
            1.0  * r_hip_pos         +   # hip_pos
            5.0  * r_dof_pos_limits  +   # dof_pos_limits
            1.0  * r_collision       +   # collision
            0.18 * r_contact         +   # contact
            20.0 * r_swing_height    +   # feet_swing_height
            0.2  * r_contact_no_vel  +   # contact_no_vel
            0.15 * r_alive           +   # alive
            1.0  * r_dof_acc         +   # dof_acc
            1.0  * r_act_rate            # action_rate
        )

    # ------------------------------------------------------------------
    def _check_termination(self, contact_forces):
        quat  = self.robot.get_quat()
        pos   = self.robot.get_pos()
        proj_g = self._q_inv_rotate(
            quat,
            torch.tensor([[0., 0., -1.]], device=self.device).expand(self.num_envs, -1),
        )
        height_fail  = pos[:, 2] < 0.5
        tilt_fail    = proj_g[:, 2] > -0.5
        timeout      = self.episode_len_buf >= self.max_episode_length
        # Terminate if pelvis touches ground
        pelvis_contact = torch.norm(contact_forces[:, self.pelvis_link_idx, :], dim=-1) > 1.0
        return height_fail | tilt_fail | timeout | pelvis_contact
