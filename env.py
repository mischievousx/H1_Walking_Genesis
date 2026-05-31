import os
import torch
import genesis as gs

URDF_PATH = os.path.join(os.path.dirname(__file__), 'robot/urdf/h1.urdf')

# DOF order (after 6-DOF free root joint):
# l_hip_yaw, r_hip_yaw, l_hip_roll, r_hip_roll,
# l_hip_pitch, r_hip_pitch, l_knee, r_knee, l_ankle, r_ankle
_DEFAULT_DOF_POS = torch.tensor([
    0.,  0.,    # hip yaw  L, R
    0.,  0.,    # hip roll L, R
   -0.1,-0.1,  # hip pitch L, R
    0.3, 0.3,  # knee L, R
   -0.2,-0.2,  # ankle L, R
])
_KP = torch.tensor([150,150,150,150,150,150,200,200,40,40], dtype=torch.float32)
_KD = torch.tensor([  2,  2,  2,  2,  2,  2,  4,  4, 2, 2], dtype=torch.float32)


class H1WalkingEnv:
    num_obs = 42
    num_actions = 10

    def __init__(self, num_envs=4096, sim_dt=0.005, control_decimation=4, device='cuda'):
        self.num_envs = num_envs
        self.device = device
        self.sim_dt = sim_dt
        self.control_decimation = control_decimation
        self.dt = sim_dt * control_decimation      # 0.02 s = 50 Hz policy
        self.max_episode_length = 1000             # 20 s per episode

        self.default_dof_pos = _DEFAULT_DOF_POS.to(device)
        self.motor_dofs = torch.arange(6, 16, device=device)

        self.commands         = torch.zeros(num_envs, 3, device=device)
        self.episode_len_buf  = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.last_action      = torch.zeros(num_envs, self.num_actions, device=device)
        self.last_dof_vel     = torch.zeros(num_envs, self.num_actions, device=device)

        # Foot air-time tracking: (n_envs, 2)  [left, right]
        self.foot_air_time    = torch.zeros(num_envs, 2, device=device)
        self.last_foot_contact = torch.ones(num_envs, 2, dtype=torch.bool, device=device)

        # Accumulated yaw heading for heading-tracking reward
        self.heading_yaw = torch.zeros(num_envs, device=device)

        self._build_scene()

    # ------------------------------------------------------------------
    def _build_scene(self):
        gs.init(backend=gs.cuda)
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

        # Foot link indices for contact detection
        self.foot_link_idx = [
            self.robot.get_link('left_ankle_link').idx_local,
            self.robot.get_link('right_ankle_link').idx_local,
        ]

    # ------------------------------------------------------------------
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)

        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 2] = 1.05
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0   # [w,x,y,z] identity = upright

        self.robot.set_pos(pos, envs_idx=env_ids)
        self.robot.set_quat(quat, envs_idx=env_ids)
        self.robot.set_dofs_position(
            self.default_dof_pos.unsqueeze(0).expand(n, -1),
            self.motor_dofs, envs_idx=env_ids,
        )
        self.robot.zero_all_dofs_velocity(envs_idx=env_ids)

        self.episode_len_buf[env_ids]    = 0
        self.last_action[env_ids]        = 0
        self.last_dof_vel[env_ids]       = 0
        self.foot_air_time[env_ids]      = 0
        self.last_foot_contact[env_ids]  = True
        self.heading_yaw[env_ids]        = 0
        self._sample_commands(env_ids)

        return self._get_obs()

    def _sample_commands(self, env_ids):
        n = len(env_ids)
        vx  = torch.rand(n, device=self.device) * 2.0 - 0.5    # -0.5 ~ 1.5 m/s
        vy  = torch.rand(n, device=self.device) * 0.6 - 0.3    # -0.3 ~ 0.3 m/s
        yaw = torch.rand(n, device=self.device) * 1.0 - 0.5    # -0.5 ~ 0.5 rad/s
        self.commands[env_ids] = torch.stack([vx, vy, yaw], dim=1)
        # 25 % stand-still + 40 % pure-forward (vy=0, yaw=0)  ← more straight-line training
        r = torch.rand(n, device=self.device)
        self.commands[env_ids[r < 0.25]]  = 0.
        fwd = (r >= 0.25) & (r < 0.65)
        self.commands[env_ids[fwd], 1]    = 0.
        self.commands[env_ids[fwd], 2]    = 0.

    # ------------------------------------------------------------------
    def step(self, actions):
        actions = torch.clamp(actions, -1., 1.)
        target  = self.default_dof_pos + actions * 0.35   # action_scale 0.25 → 0.35

        dof_vel_prev = self.robot.get_dofs_velocity(self.motor_dofs).clone()

        for _ in range(self.control_decimation):
            self.robot.control_dofs_position(target, self.motor_dofs)
            self.scene.step()

        self.episode_len_buf += 1

        # ── Foot contact detection ────────────────────────────────────────
        contact_forces = self.robot.get_links_net_contact_force()  # (n, n_links, 3)
        foot_contact = torch.stack([
            torch.norm(contact_forces[:, self.foot_link_idx[0], :], dim=-1) > 1.0,
            torch.norm(contact_forces[:, self.foot_link_idx[1], :], dim=-1) > 1.0,
        ], dim=1)  # (n_envs, 2)  True = on ground

        # Accumulate air time while foot is off ground (reset AFTER rewards to avoid zeroing just-landed feet)
        self.foot_air_time += self.dt

        dof_vel_now = self.robot.get_dofs_velocity(self.motor_dofs)
        dof_acc     = (dof_vel_now - dof_vel_prev) / self.dt

        # Accumulate heading yaw angle for heading-tracking reward
        ang_vel_z = self._q_inv_rotate(self.robot.get_quat(), self.robot.get_ang())[:, 2]
        self.heading_yaw += ang_vel_z * self.dt
        # Wrap to [-π, π] to prevent unbounded growth and reward explosion
        self.heading_yaw.clamp_(-3.14159, 3.14159)

        rew  = self._compute_rewards(actions, dof_acc, foot_contact)
        done = self._check_termination()

        # Reset air time after rewards so just-landed feet carry their accumulated time into the reward
        self.foot_air_time[foot_contact] = 0.

        self.last_action       = actions.clone()
        self.last_dof_vel      = dof_vel_now.clone()
        self.last_foot_contact = foot_contact.clone()

        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self.reset(done_ids)

        obs = self._get_obs()
        return obs, rew, done

    # ------------------------------------------------------------------
    def _get_obs(self):
        quat = self.robot.get_quat()
        lin_vel  = self._q_inv_rotate(quat, self.robot.get_vel())
        ang_vel  = self._q_inv_rotate(quat, self.robot.get_ang())
        proj_g   = self._q_inv_rotate(
            quat,
            torch.tensor([[0., 0., -1.]], device=self.device).expand(self.num_envs, -1),
        )
        dof_pos = self.robot.get_dofs_position(self.motor_dofs) - self.default_dof_pos
        dof_vel = self.robot.get_dofs_velocity(self.motor_dofs)

        cmd_scale = torch.tensor([2., 2., 0.25], device=self.device)
        obs = torch.cat([
            lin_vel,                        # 3
            ang_vel  * 0.25,               # 3
            proj_g,                         # 3
            self.commands * cmd_scale,      # 3
            dof_pos,                        # 10
            dof_vel  * 0.05,               # 10
            self.last_action,              # 10
        ], dim=-1)                          # = 42
        return torch.clamp(obs, -5., 5.)

    def _q_inv_rotate(self, q, v):
        """Rotate v by q^{-1}. q: (n,4) [w,x,y,z], v: (n,3)."""
        qw  = q[:, 0]
        qv  = q[:, 1:]
        a = v * (2. * qw**2 - 1.).unsqueeze(-1)
        b = torch.cross(qv, v, dim=1) * (2. * qw).unsqueeze(-1)
        c = qv * torch.sum(qv * v, dim=1, keepdim=True) * 2.
        return a - b + c

    # ------------------------------------------------------------------
    def _compute_rewards(self, actions, dof_acc, foot_contact):
        quat    = self.robot.get_quat()
        lin_vel = self._q_inv_rotate(quat, self.robot.get_vel())
        ang_vel = self._q_inv_rotate(quat, self.robot.get_ang())

        # ── Velocity tracking ─────────────────────────────────────────────
        lin_err = torch.sum((self.commands[:, :2] - lin_vel[:, :2])**2, dim=1)
        ang_err = (self.commands[:, 2] - ang_vel[:, 2])**2
        r_lin   = torch.exp(-lin_err / 0.25)
        r_ang   = torch.exp(-ang_err / 0.10)   # sharper: 0.25 → 0.10, more sensitive to yaw drift

        # ── Straight-line: penalise lateral vel when vy_cmd ≈ 0 ──────────
        vy_cmd_small  = self.commands[:, 1].abs() < 0.1
        yaw_cmd_small = self.commands[:, 2].abs() < 0.1
        r_lat = -lin_vel[:, 1]**2 * vy_cmd_small.float()

        # ── Yaw rate zero penalty: direct L2 on current yaw rate when cmd ≈ 0 ──
        # Targets instantaneous drift that exp-based r_ang is insensitive to at small values
        r_yaw_zero = -ang_vel[:, 2]**2 * yaw_cmd_small.float()

        # ── Heading tracking: penalise accumulated yaw when yaw_cmd ≈ 0 ──
        # Clamped to max=4 to prevent reward explosion late in long episodes
        r_heading = -(self.heading_yaw**2).clamp(max=4.0) * yaw_cmd_small.float()

        # ── Foot air-time: reward bigger strides ─────────────────────────
        # Give reward when foot lands after being in air > 0.1 s
        just_landed   = self.last_foot_contact & ~foot_contact  # was off, now on
        # actually: reward when foot that WAS in air just contacted ground
        just_landed   = (~self.last_foot_contact) & foot_contact
        air_reward    = torch.clamp(self.foot_air_time, 0., 0.5)  # cap at 0.5 s
        r_air_time    = torch.sum(air_reward * just_landed.float(), dim=1)

        # ── Stability ─────────────────────────────────────────────────────
        r_roll_pitch = -torch.sum(ang_vel[:, :2]**2, dim=1)
        r_height     = -(self.robot.get_pos()[:, 2] - 1.05)**2

        # ── Smoothness ────────────────────────────────────────────────────
        r_dof_acc  = -torch.sum(dof_acc**2, dim=1) * 2.5e-7
        r_act_rate = -torch.sum((actions - self.last_action)**2, dim=1) * 0.01

        return (
            1.0  * r_lin        +   # forward velocity tracking
            2.5  * r_ang        +   # yaw rate tracking (↑ from 1.5)
            3.0  * r_lat        +   # lateral velocity penalty (↑ from 2.0)
            2.0  * r_yaw_zero   +   # direct yaw rate L2 penalty (new)
            1.5  * r_heading    +   # accumulated heading penalty (↑ from 1.0, now clamped)
            2.0  * r_air_time   +   # bigger strides
            0.05 * r_roll_pitch +   # stay upright
            0.5  * r_height     +   # maintain height
            1.0  * r_dof_acc    +   # smooth joints
            1.0  * r_act_rate       # smooth actions
        )

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
