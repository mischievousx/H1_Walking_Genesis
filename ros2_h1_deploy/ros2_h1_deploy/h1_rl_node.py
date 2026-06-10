"""
H1 RL Controller Node
=====================
ROS2 node that deploys the trained LSTM policy onto the real Unitree H1.

Communication:
  DDS  (unitree_sdk2py) ← LowState  (500 Hz, robot → node)
  DDS  (unitree_sdk2py) → LowCmd    (50 Hz,  node → robot)
  ROS2 ← /cmd_vel (geometry_msgs/Twist, high-level velocity command)

State machine:
  ZERO_TORQUE  →(start button)→  MOVE_TO_DEFAULT
  MOVE_TO_DEFAULT  →(2 s)→  HOLD_DEFAULT
  HOLD_DEFAULT  →(A button)→  RL_CONTROL
  RL_CONTROL  →(select button / KeyboardInterrupt)→  DAMPING

Usage:
  ros2 run ros2_h1_deploy h1_rl_node --ros-args -p net:=eth0 -p checkpoint:=/path/to/h1_walk_best.pt
"""

import sys
import os
import math
import time
import threading
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ── unitree SDK ──────────────────────────────────────────────────────────────
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from scipy.spatial.transform import Rotation as R

# ── project policy ───────────────────────────────────────────────────────────
# Add the training project to path so we can import ppo.py
_PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'h1_genesis_train')
sys.path.insert(0, os.path.abspath(_PROJECT_DIR))
from ppo import ActorCriticRecurrent  # noqa: E402


# ── hardware constants ───────────────────────────────────────────────────────
# Leg joint → motor index mapping in LowState.motor_state[]
# Order matches training obs: [L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee, L_ankle,
#                               R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee, R_ankle]
LEG_MOTOR_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Waist yaw motor index (used for IMU frame transform; H1 torso IMU)
WAIST_YAW_IDX = 12  # verify against your H1 firmware

# Default standing pose (rad) — must match training DEFAULT_DOF
DEFAULT_DOF = np.array([0., 0., 0., 0., -0.1,
                        -0.1, 0.3, 0.3, -0.2, -0.2], dtype=np.float32)

KP = np.array([150, 150, 150, 150, 150, 150, 200, 200, 40, 40], dtype=np.float32)
KD = np.array([  2,   2,   2,   2,   2,   2,   4,   4,  2,  2], dtype=np.float32)

NUM_ACTIONS = 10
NUM_OBS     = 41
CONTROL_DT  = 0.02        # 50 Hz
ACTION_SCALE = 0.25
PHASE_PERIOD = 0.8        # seconds

# Obs scaling (must match env.py training scales)
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE     = np.array([2., 2., 0.25], dtype=np.float32)  # [vx, vy, yaw]

# Remote controller button map (wireless_remote byte indices)
class KeyMap:
    start  = 3   # byte 3 bit 4
    A      = 2   # byte 2 bit 0
    select = 3   # byte 3 bit 5


class ControlState:
    ZERO_TORQUE    = 0
    MOVE_TO_DEFAULT = 1
    HOLD_DEFAULT   = 2
    RL_CONTROL     = 3
    DAMPING        = 4


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_gravity_orientation(quat_wxyz: np.ndarray) -> np.ndarray:
    """Project gravity [0,0,-1] into the pelvis body frame."""
    qw, qx, qy, qz = quat_wxyz
    return np.array([
         2 * (-qz * qx + qw * qy),
        -2 * ( qz * qy + qw * qx),
         1 - 2 * (qw * qw + qz * qz),
    ], dtype=np.float32)


def _transform_imu(waist_yaw: float, waist_yaw_omega: float,
                   imu_quat_wxyz: np.ndarray, imu_omega: np.ndarray):
    """Rotate torso IMU data into the pelvis frame (H1-specific)."""
    Rz = R.from_euler('z', waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat_wxyz[1], imu_quat_wxyz[2],
                           imu_quat_wxyz[3], imu_quat_wxyz[0]]).as_matrix()
    R_pelvis = R_torso @ Rz.T
    omega_pelvis = Rz @ imu_omega - np.array([0., 0., waist_yaw_omega])
    quat_xyzw = R.from_matrix(R_pelvis).as_quat()
    return quat_xyzw[[3, 0, 1, 2]], omega_pelvis  # back to wxyz


def _remote_button(wireless_remote: list, key: int) -> bool:
    """Parse wireless_remote byte array for button presses."""
    if key == KeyMap.start:
        return bool((wireless_remote[3] >> 4) & 1)
    if key == KeyMap.A:
        return bool((wireless_remote[2] >> 0) & 1)
    if key == KeyMap.select:
        return bool((wireless_remote[3] >> 5) & 1)
    return False


# ── main node ────────────────────────────────────────────────────────────────

class H1RLNode(Node):
    def __init__(self):
        super().__init__('h1_rl_node')

        # ── parameters ──────────────────────────────────────────────────────
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('net', 'eth0')

        checkpoint_path = self.get_parameter('checkpoint').get_parameter_value().string_value
        net_iface       = self.get_parameter('net').get_parameter_value().string_value

        if not checkpoint_path:
            self.get_logger().fatal('Parameter "checkpoint" is required.')
            raise RuntimeError('No checkpoint specified.')

        # ── policy ──────────────────────────────────────────────────────────
        self.device = 'cpu'  # real robot: use CPU for deterministic timing
        self.ac = ActorCriticRecurrent(NUM_OBS, NUM_ACTIONS,
                                       num_privileged_obs=NUM_OBS + 3).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.ac.load_state_dict(ckpt['model'])
        self.ac.eval()
        self.get_logger().info(f'Loaded policy: {checkpoint_path}  '
                               f'(step {ckpt.get("total_steps", "?")})')

        # obs normalizer (optional, v5 checkpoints)
        self.obs_norm = None
        ckpt_dir = os.path.dirname(checkpoint_path)
        for candidate in [os.path.join(ckpt_dir, 'obs_norm_best.pt'),
                          os.path.join(ckpt_dir, 'obs_norm.pt')]:
            if os.path.exists(candidate):
                nd = torch.load(candidate, map_location=self.device, weights_only=True)
                if 'obs' in nd:
                    self.obs_norm = {
                        'mean': nd['obs']['mean'].to(self.device),
                        'var':  nd['obs']['var'].to(self.device),
                    }
                    self.get_logger().info(f'Loaded obs normalizer: {candidate}')
                break

        # ── runtime state ────────────────────────────────────────────────────
        self.state      = ControlState.ZERO_TORQUE
        self.lock       = threading.Lock()
        self.low_state  = unitree_go_msg_dds__LowState_()
        self.low_cmd    = unitree_go_msg_dds__LowCmd_()
        self._init_lowcmd()

        self.action     = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.cmd        = np.zeros(3, dtype=np.float32)   # [vx, vy, yaw_rate]
        self.phase      = 0.0
        self.counter    = 0
        self.hidden     = self.ac.init_hidden(1, self.device)

        # move_to_default bookkeeping
        self._move_start_time   = None
        self._move_init_dof_pos = None
        self._move_total_steps  = int(2.0 / CONTROL_DT)
        self._move_step         = 0

        # ── DDS setup ───────────────────────────────────────────────────────
        ChannelFactoryInitialize(0, net_iface)

        self._lowcmd_pub = ChannelPublisher('rt/lowcmd', LowCmdGo)
        self._lowcmd_pub.Init()

        self._lowstate_sub = ChannelSubscriber('rt/lowstate', LowStateGo)
        self._lowstate_sub.Init(self._lowstate_callback, 10)

        self.get_logger().info('Waiting for LowState...')
        while self.low_state.tick == 0:
            time.sleep(0.01)
        self.get_logger().info('Connected to robot.')

        # ── ROS2 interfaces ──────────────────────────────────────────────────
        self._cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, 10)

        self._timer = self.create_timer(CONTROL_DT, self._control_loop)

        self.get_logger().info(
            'H1 RL node ready.  State: ZERO_TORQUE\n'
            '  → Press START  to move to default pose\n'
            '  → Press A      to start RL control\n'
            '  → Press SELECT to stop (damping)')

    # ── DDS callbacks ────────────────────────────────────────────────────────

    def _lowstate_callback(self, msg: LowStateGo):
        with self.lock:
            self.low_state = msg

    def _cmd_vel_callback(self, msg: Twist):
        self.cmd[0] = float(msg.linear.x)
        self.cmd[1] = float(msg.linear.y)
        self.cmd[2] = float(msg.angular.z)

    # ── LowCmd helpers ───────────────────────────────────────────────────────

    def _init_lowcmd(self):
        PosStopF = 2.146e9
        VelStopF = 16000.0
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(len(self.low_cmd.motor_cmd)):
            self.low_cmd.motor_cmd[i].mode = 0x0A
            self.low_cmd.motor_cmd[i].q    = PosStopF
            self.low_cmd.motor_cmd[i].qd   = VelStopF
            self.low_cmd.motor_cmd[i].kp   = 0.
            self.low_cmd.motor_cmd[i].kd   = 0.
            self.low_cmd.motor_cmd[i].tau  = 0.

    def _send_cmd(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self._lowcmd_pub.Write(self.low_cmd)

    def _set_zero_torque(self):
        for i in range(len(self.low_cmd.motor_cmd)):
            self.low_cmd.motor_cmd[i].q   = 0.
            self.low_cmd.motor_cmd[i].qd  = 0.
            self.low_cmd.motor_cmd[i].kp  = 0.
            self.low_cmd.motor_cmd[i].kd  = 0.
            self.low_cmd.motor_cmd[i].tau = 0.

    def _set_damping(self):
        for i in range(len(self.low_cmd.motor_cmd)):
            self.low_cmd.motor_cmd[i].q   = 0.
            self.low_cmd.motor_cmd[i].qd  = 0.
            self.low_cmd.motor_cmd[i].kp  = 0.
            self.low_cmd.motor_cmd[i].kd  = 8.
            self.low_cmd.motor_cmd[i].tau = 0.

    # ── observation builder ──────────────────────────────────────────────────

    def _build_obs(self) -> torch.Tensor:
        with self.lock:
            ls = self.low_state

        # joint positions & velocities
        qj  = np.array([ls.motor_state[i].q  for i in LEG_MOTOR_IDX], dtype=np.float32)
        dqj = np.array([ls.motor_state[i].dq for i in LEG_MOTOR_IDX], dtype=np.float32)

        # IMU (H1 torso IMU → transform to pelvis frame)
        imu_quat = np.array(ls.imu_state.quaternion, dtype=np.float32)  # [w, x, y, z]
        imu_gyro = np.array(ls.imu_state.gyroscope,  dtype=np.float32)
        waist_yaw       = float(ls.motor_state[WAIST_YAW_IDX].q)
        waist_yaw_omega = float(ls.motor_state[WAIST_YAW_IDX].dq)
        pelvis_quat, pelvis_omega = _transform_imu(
            waist_yaw, waist_yaw_omega, imu_quat, imu_gyro)

        ang_vel = pelvis_omega.astype(np.float32)
        proj_g  = _get_gravity_orientation(pelvis_quat)

        # phase clock
        sin_ph = math.sin(2. * math.pi * self.phase)
        cos_ph = math.cos(2. * math.pi * self.phase)

        obs = np.concatenate([
            ang_vel * ANG_VEL_SCALE,                   # 3
            proj_g,                                     # 3
            self.cmd * CMD_SCALE,                       # 3
            (qj - DEFAULT_DOF) * DOF_POS_SCALE,        # 10
            dqj * DOF_VEL_SCALE,                       # 10
            self.action,                                # 10
            [sin_ph, cos_ph],                           # 2
        ]).astype(np.float32)                           # = 41

        obs = np.clip(obs, -5., 5.)
        t = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        if self.obs_norm is not None:
            t = torch.clamp(
                (t - self.obs_norm['mean']) / torch.sqrt(self.obs_norm['var'] + 1e-4),
                -10., 10.)
        return t

    # ── control loop (50 Hz ROS2 timer) ─────────────────────────────────────

    def _control_loop(self):
        with self.lock:
            wireless = list(self.low_state.wireless_remote)

        btn_start  = _remote_button(wireless, KeyMap.start)
        btn_a      = _remote_button(wireless, KeyMap.A)
        btn_select = _remote_button(wireless, KeyMap.select)

        # ── state transitions ────────────────────────────────────────────────
        if btn_select and self.state == ControlState.RL_CONTROL:
            self.get_logger().info('SELECT pressed → DAMPING')
            self.state = ControlState.DAMPING

        if self.state == ControlState.ZERO_TORQUE:
            if btn_start:
                self.get_logger().info('START pressed → MOVE_TO_DEFAULT')
                with self.lock:
                    self._move_init_dof_pos = np.array(
                        [self.low_state.motor_state[i].q for i in LEG_MOTOR_IDX],
                        dtype=np.float32)
                self._move_step = 0
                self.state = ControlState.MOVE_TO_DEFAULT
            else:
                self._set_zero_torque()
                self._send_cmd()
                return

        if self.state == ControlState.MOVE_TO_DEFAULT:
            alpha = min(self._move_step / self._move_total_steps, 1.0)
            target = self._move_init_dof_pos * (1. - alpha) + DEFAULT_DOF * alpha
            for j, motor_idx in enumerate(LEG_MOTOR_IDX):
                self.low_cmd.motor_cmd[motor_idx].q   = float(target[j])
                self.low_cmd.motor_cmd[motor_idx].qd  = 0.
                self.low_cmd.motor_cmd[motor_idx].kp  = float(KP[j])
                self.low_cmd.motor_cmd[motor_idx].kd  = float(KD[j])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.
            self._send_cmd()
            self._move_step += 1
            if self._move_step >= self._move_total_steps:
                self.get_logger().info('Default pose reached → HOLD_DEFAULT')
                self.state = ControlState.HOLD_DEFAULT
            return

        if self.state == ControlState.HOLD_DEFAULT:
            for j, motor_idx in enumerate(LEG_MOTOR_IDX):
                self.low_cmd.motor_cmd[motor_idx].q   = float(DEFAULT_DOF[j])
                self.low_cmd.motor_cmd[motor_idx].qd  = 0.
                self.low_cmd.motor_cmd[motor_idx].kp  = float(KP[j])
                self.low_cmd.motor_cmd[motor_idx].kd  = float(KD[j])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.
            self._send_cmd()
            if btn_a:
                self.get_logger().info('A pressed → RL_CONTROL')
                self.hidden  = self.ac.init_hidden(1, self.device)
                self.phase   = 0.0
                self.counter = 0
                self.action  = np.zeros(NUM_ACTIONS, dtype=np.float32)
                self.state   = ControlState.RL_CONTROL
            return

        if self.state == ControlState.RL_CONTROL:
            obs = self._build_obs()
            with torch.no_grad():
                action_t, self.hidden = self.ac.act_deterministic(obs, self.hidden)
            action_t = torch.clamp(action_t, -1., 1.)
            self.action = action_t.squeeze(0).cpu().numpy()

            target_dof = DEFAULT_DOF + self.action * ACTION_SCALE
            for j, motor_idx in enumerate(LEG_MOTOR_IDX):
                self.low_cmd.motor_cmd[motor_idx].q   = float(target_dof[j])
                self.low_cmd.motor_cmd[motor_idx].qd  = 0.
                self.low_cmd.motor_cmd[motor_idx].kp  = float(KP[j])
                self.low_cmd.motor_cmd[motor_idx].kd  = float(KD[j])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.
            self._send_cmd()

            self.phase    = (self.phase + CONTROL_DT / PHASE_PERIOD) % 1.0
            self.counter += 1
            return

        if self.state == ControlState.DAMPING:
            self._set_damping()
            self._send_cmd()


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = H1RLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt → DAMPING')
    finally:
        node._set_damping()
        node._send_cmd()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
