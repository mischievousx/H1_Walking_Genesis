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

RESAMPLE_INTERVAL = 500   # resample commands every 10 s (500 steps × 0.02 s)


class H1WalkingEnv:
    num_obs     = 41   # ang_vel(3)+proj_g(3)+cmd(3)+dof_pos(10)+dof_vel(10)+action(10)+phase(2)
    num_actions = 10

    def __init__(self, num_envs=4096, sim_dt=0.005, control_decimation=4, device='cuda'):
        self.num_envs           = num_envs
        self.device             = device
        self.sim_dt             = sim_dt
        self.control_decimation = control_decimation
        self.dt                 = sim_dt * control_decimation   # 0.02 s = 50 Hz
        self.max_episode_length = 1000                          # 20 s

        self.default_dof_pos = _DEFAULT_DOF_POS.to(device)
        self.motor_dofs      = torch.arange(6, 16, device=device)

        self.commands          = torch.zeros(num_envs, 3, device=device)
        self.heading_target    = torch.zeros(num_envs, device=device)
        self.episode_len_buf   = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.last_action       = torch.zeros(num_envs, self.num_actions, device=device)
        self.last_dof_vel      = torch.zeros(num_envs, self.num_actions, device=device)
        self.last_dof_acc      = torch.zeros(num_envs, self.num_actions, device=device)

        self.foot_air_time     = torch.zeros(num_envs, 2, device=device)
        self.last_foot_contact = torch.ones(num_envs, 2, dtype=torch.bool, device=device)

        # Gait phase [0, 1), period = 0.8 s, right leg leads by 0.5
        self.phase = torch.zeros(num_envs, device=device)

        self._build_scene()

    # ------------------------------------------------------------------
    def _build_scene(self):
        gs.init(backend=gs.cuda, logging_level='error')
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=1),
            show_viewer=False,
            show_FPS=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
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

        self.episode_len_buf[env_ids]   = 0
        self.last_action[env_ids]       = 0
        self.last_dof_vel[env_ids]      = 0
        self.last_dof_acc[env_ids]      = 0
        self.foot_air_time[env_ids]     = 0
        self.last_foot_contact[env_ids] = True
        self.phase[env_ids]             = torch.rand(n, device=self.device)
        self._sample_commands(env_ids)

        return self._get_obs()

    # ------------------------------------------------------------------
    def _sample_commands(self, env_ids):
        n = len(env_ids)
        vx = torch.rand(n, device=self.device) * 2.0 - 1.0   # -1.0 ~ 1.0 m/s
        vy = torch.rand(n, device=self.device) * 1.0 - 0.5   # -0.5 ~ 0.5 m/s
        self.commands[env_ids, 0] = vx
        self.commands[env_ids, 1] = vy
        # Sample target heading; yaw cmd is derived each step from heading error
        self.heading_target[env_ids] = (
            torch.rand(n, device=self.device) * 2 * 3.14159 - 3.14159
        )
        # Zero small velocity commands (|vxy| < 0.2 m/s → stand still)
        small = torch.norm(self.commands[env_ids, :2], dim=1) < 0.2
        self.commands[env_ids[small], :2] = 0.
        self.heading_target[env_ids[small]] = 0.

    def _update_heading_command(self):
        """Compute yaw-rate command from heading error each step."""
        quat = self.robot.get_quat()   # (n, 4) [w,x,y,z]
        qw, qx, qy, qz = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        heading = torch.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        err = self.heading_target - heading
        err = ((err + 3.14159) % (2 * 3.14159)) - 3.14159   # wrap to [-π, π]
        self.commands[:, 2] = torch.clamp(0.5 * err, -1., 1.)
        # Keep yaw cmd = 0 when velocity command is zero
        no_vel = torch.norm(self.commands[:, :2], dim=1) < 0.1
        self.commands[no_vel, 2] = 0.

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

        # Periodically resample commands (every 10 s)
        resample_ids = (self.episode_len_buf % RESAMPLE_INTERVAL == 0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._sample_commands(resample_ids)

        # Update yaw command from heading error
        self._update_heading_command()

        # ── Foot contact ──────────────────────────────────────────────────
        contact_forces = self.robot.get_links_net_contact_force()
        foot_contact = torch.stack([
            torch.norm(contact_forces[:, self.foot_link_idx[0], :], dim=-1) > 1.0,
            torch.norm(contact_forces[:, self.foot_link_idx[1], :], dim=-1) > 1.0,
        ], dim=1)

        self.foot_air_time += self.dt

        dof_vel_now = self.robot.get_dofs_velocity(self.motor_dofs)
        dof_acc     = (dof_vel_now - dof_vel_prev) / self.dt
        dof_jerk    = (dof_acc - self.last_dof_acc) / self.dt

        rew  = self._compute_rewards(actions, dof_acc, dof_jerk, foot_contact)
        # Clip total reward to ≥ 0 (avoids purely negative signal destabilising training)
        rew  = torch.clamp(rew, min=0.)
        done = self._check_termination()

        self.foot_air_time[foot_contact] = 0.
        self.last_action       = actions.clone()
        self.last_dof_vel      = dof_vel_now.clone()
        self.last_foot_contact = foot_contact.clone()
        self.last_dof_acc      = dof_acc.clone()

        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self.reset(done_ids)

        return self._get_obs(), rew, done

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
            ang_vel * 0.25,            # 3
            proj_g,                    # 3
            self.commands * cmd_scale, # 3
            dof_pos,                   # 10
            dof_vel * 0.05,            # 10
            self.last_action,          # 10
            sin_phase,                 # 1
            cos_phase,                 # 1
        ], dim=-1)                     # = 41
        return torch.clamp(obs, -5., 5.)

    def _q_inv_rotate(self, q, v):
        qw = q[:, 0]; qv = q[:, 1:]
        a  = v * (2. * qw**2 - 1.).unsqueeze(-1)
        b  = torch.cross(qv, v, dim=1) * (2. * qw).unsqueeze(-1)
        c  = qv * torch.sum(qv * v, dim=1, keepdim=True) * 2.
        return a - b + c

    # ------------------------------------------------------------------
    def _compute_rewards(self, actions, dof_acc, dof_jerk, foot_contact):
        quat    = self.robot.get_quat()
        lin_vel = self._q_inv_rotate(quat, self.robot.get_vel())
        ang_vel = self._q_inv_rotate(quat, self.robot.get_ang())
        proj_g  = self._q_inv_rotate(
            quat,
            torch.tensor([[0., 0., -1.]], device=self.device).expand(self.num_envs, -1),
        )

        # ── Velocity tracking ─────────────────────────────────────────────
        lin_err = torch.sum((self.commands[:, :2] - lin_vel[:, :2])**2, dim=1)
        ang_err = (self.commands[:, 2] - ang_vel[:, 2])**2
        r_lin   = torch.exp(-lin_err / 0.1)
        r_ang   = torch.exp(-ang_err / 0.1)

        # ── Body motion penalties ──────────────────────────────────────────
        r_lin_vel_z  = -lin_vel[:, 2]**2
        r_ang_vel_xy = -torch.sum(ang_vel[:, :2]**2, dim=1)
        r_orientation = -torch.sum(proj_g[:, :2]**2, dim=1)
        r_height      = -(self.robot.get_pos()[:, 2] - 1.05)**2

        # ── Joint penalties ────────────────────────────────────────────────
        dof_pos   = self.robot.get_dofs_position(self.motor_dofs)
        r_hip_pos = -torch.sum(dof_pos[:, [0, 1, 2, 3]]**2, dim=1)

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
        r_jerk     = -torch.sum(dof_jerk**2, dim=1) * 2.5e-11
        r_act_rate = -torch.sum((actions - self.last_action)**2, dim=1) * 0.01

        return (
            3.0  * r_lin             +   # tracking lin vel
            1.5  * r_ang             +   # tracking ang vel (heading-derived)
            2.0  * r_lin_vel_z       +   # penalise vertical body vel
            0.05 * r_ang_vel_xy      +   # penalise roll/pitch angular vel
            1.0  * r_orientation     +   # penalise body tilt
            10.0 * r_height          +   # maintain height
            1.0  * r_hip_pos         +   # keep hips centred
            0.18 * r_contact         +   # gait phase contact timing
            20.0 * r_swing_height    +   # foot swing height during swing
            0.2  * r_contact_no_vel  +   # no foot sliding on contact
            0.15 * r_alive           +   # alive bonus
            1.0  * r_dof_acc         +   # acceleration smoothness
            1.0  * r_jerk            +   # jerk smoothness, suppresses vibration
            1.0  * r_act_rate            # smooth actions
        )

    # ------------------------------------------------------------------
    def _check_termination(self):
        quat  = self.robot.get_quat()
        pos   = self.robot.get_pos()
        proj_g = self._q_inv_rotate(
            quat,
            torch.tensor([[0., 0., -1.]], device=self.device).expand(self.num_envs, -1),
        )
        height_fail = pos[:, 2] < 0.5
        tilt_fail   = proj_g[:, 2] > -0.5
        timeout     = self.episode_len_buf >= self.max_episode_length
        return height_fail | tilt_fail | timeout
