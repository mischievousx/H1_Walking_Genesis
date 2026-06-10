"""
ROS2 closed-loop Genesis simulation
=====================================
Closed-loop architecture through ROS2 topics:

  Genesis sim  →  /imu (sensor_msgs/Imu, actual robot orientation)
                       ↓
              DemoSequenceNode  (heading error → yaw_rate, P-control)
                       ↓  /cmd_vel (geometry_msgs/Twist)
              Genesis sim  →  RL policy  →  joint targets  →  LowCmd

Compared to play_ros2_path_openloop.py (open-loop heading integrator),
this version closes the outer heading loop with real robot orientation.

Usage
-----
  source /opt/ros/humble/setup.bash
  python3 play_ros2_path.py \\
      --checkpoint checkpoints/checkpoints_v4_r48s_acc5x/h1_walk_best.pt \\
      --out videos/ros2_path_closed_acc5x.mp4
"""

import argparse
import math
import os
import queue
import threading

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import quadrants as _qd
_qd_init_orig = _qd.init
def _qd_init_patched(**kwargs):
    kwargs.setdefault('device_memory_GB', 16.0)
    return _qd_init_orig(**kwargs)
_qd.init = _qd_init_patched

import genesis as gs

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

from ppo import ActorCriticRecurrent

# ── constants ────────────────────────────────────────────────────────────────
URDF_PATH    = os.path.join(os.path.dirname(__file__), 'robot/urdf/h1.urdf')
NUM_OBS      = 41
NUM_ACTIONS  = 10
DEFAULT_DOF  = torch.tensor([0., 0., 0., 0., -0.1, -0.1, 0.3, 0.3, -0.2, -0.2])
MOTOR_DOFS   = list(range(6, 16))
KP = torch.tensor([150, 150, 150, 150, 150, 150, 200, 200, 40, 40], dtype=torch.float32)
KD = torch.tensor([  2,   2,   2,   2,   2,   2,   4,   4,  2,  2], dtype=torch.float32)

CONTROL_DT   = 0.02
PHASE_PERIOD = 0.8
VX_RAMP_RATE = 0.02
YAW_GAIN     = 1.0

# Deterministic route — mirrors demo_sequence.py's PHASES:
# forward → left turn → forward → right turn → forward → stop.
# Each entry: (steps, vx, vy, heading_target_rad, label)
FIXED_ROUTE = [
    (200, 0.5, 0.0, 0.0,        'Forward'),
    (300, 0.4, 0.0, math.pi/2,  'Left turn'),
    (200, 0.5, 0.0, math.pi/2,  'Forward'),
    (300, 0.4, 0.0, 0.0,        'Right turn'),
    (150, 0.5, 0.0, 0.0,        'Forward'),
    (100, 0.0, 0.0, 0.0,        'Stop'),
]


def generate_fixed_route():
    """Return the deterministic square-loop command sequence (no RNG —
    same route every run, useful for repeatable path-tracking evaluation)."""
    return list(FIXED_ROUTE)


# ── sim helpers ───────────────────────────────────────────────────────────────

def quat_rotate_inverse(q, v):
    qw = q[:, 0]; qv = q[:, 1:]
    a = v * (2. * qw**2 - 1.).unsqueeze(-1)
    b = torch.cross(qv, v, dim=1) * (2. * qw).unsqueeze(-1)
    c = qv * torch.sum(qv * v, dim=1, keepdim=True) * 2.
    return a - b + c


def get_obs(robot, last_action, commands, phase, device, obs_norm=None):
    quat    = robot.get_quat()
    ang_vel = quat_rotate_inverse(quat, robot.get_ang())
    proj_g  = quat_rotate_inverse(quat, torch.tensor([[0., 0., -1.]], device=device))
    motor_dofs = torch.tensor(MOTOR_DOFS, device=device)
    dof_pos = robot.get_dofs_position(motor_dofs) - DEFAULT_DOF.to(device)
    dof_vel = robot.get_dofs_velocity(motor_dofs)
    cmd_scale = torch.tensor([2., 2., 0.25], device=device)
    sin_phase = torch.sin(2 * math.pi * phase).unsqueeze(1)
    cos_phase = torch.cos(2 * math.pi * phase).unsqueeze(1)
    obs = torch.cat([
        ang_vel * 0.25, proj_g, commands * cmd_scale,
        dof_pos, dof_vel * 0.05, last_action, sin_phase, cos_phase,
    ], dim=-1)
    obs = torch.clamp(obs, -5., 5.)
    if obs_norm is not None:
        obs = torch.clamp(
            (obs - obs_norm['mean']) / torch.sqrt(obs_norm['var'] + 1e-4),
            -10., 10.)
    return obs


def quat_to_yaw(qw, qx, qy, qz) -> float:
    """Extract yaw angle from quaternion (wxyz convention)."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def annotate(rgb: np.ndarray, label: str, vx: float,
             heading_actual: float, heading_target: float, step: int) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
    except OSError:
        font = ImageFont.load_default()
    lines = [
        f'[ROS2 closed-loop]  Phase: {label}',
        f'vx = {vx:.2f} m/s    step = {step}',
        f'heading: actual={math.degrees(heading_actual):+.1f}°  '
        f'target={math.degrees(heading_target):+.1f}°  '
        f'err={math.degrees(heading_target - heading_actual):+.1f}°',
    ]
    y = 14
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        x = img.width - tw - 28
        draw.rectangle([x - 10, y - 6, x + tw + 10, y + th + 6],
                       fill=(0, 0, 0, 180))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += th + 10
    return np.array(img)


# ── ROS2 nodes ────────────────────────────────────────────────────────────────

class ImuPublisherNode(Node):
    """
    Reads robot orientation from a shared variable (written by Genesis main thread)
    and publishes sensor_msgs/Imu to /imu at 50 Hz.
    """
    def __init__(self, imu_state: dict):
        super().__init__('genesis_imu_publisher')
        self._state = imu_state   # shared dict: {'qw':1,'qx':0,'qy':0,'qz':0}
        self._pub   = self.create_publisher(Imu, '/imu', 10)
        self.create_timer(CONTROL_DT, self._tick)

    def _tick(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'pelvis'
        msg.orientation.w = self._state['qw']
        msg.orientation.x = self._state['qx']
        msg.orientation.y = self._state['qy']
        msg.orientation.z = self._state['qz']
        # covariance: -1 means "not available" for fields we don't fill
        msg.angular_velocity_covariance[0] = -1.
        msg.linear_acceleration_covariance[0] = -1.
        self._pub.publish(msg)


class CmdVelNode(Node):
    """Subscribes to /cmd_vel and puts the latest command into cmd_queue."""
    def __init__(self, cmd_queue: queue.Queue):
        super().__init__('genesis_cmdvel_subscriber')
        self._q = cmd_queue
        self.create_subscription(Twist, '/cmd_vel', self._cb, 10)

    def _cb(self, msg: Twist):
        try:
            self._q.put_nowait((msg.linear.x, msg.linear.y, msg.angular.z))
        except queue.Full:
            try: self._q.get_nowait()
            except queue.Empty: pass
            self._q.put_nowait((msg.linear.x, msg.linear.y, msg.angular.z))


class DemoSequenceNode(Node):
    """
    Publishes /cmd_vel by stepping through a list of command segments
    (each `phases[i] = (steps, vx, vy, heading_target, label)`).
    Subscribes to /imu for CLOSED-LOOP heading feedback.
    Without /imu, falls back to open-loop yaw integration.

    NOTE on timing — sim-time vs wall-clock:
    Genesis renders frames slower than real-time (50 Hz commanded but actual
    throughput is lower due to camera rendering). A wall-clock ROS2 timer would
    therefore race ahead of the simulation and exhaust the phase list early
    (observed: ~2x desync — 1200 sim-steps took ~48 s wall-clock instead of 24 s).

    The fix mirrors how Gazebo/ROS2 solve this with /clock + use_sim_time:
    here we simply drive `step()` directly from the simulation loop — one call
    per simulated control step — instead of from a wall-clock timer. This keeps
    the publish/subscribe path 100% real ROS2 (/cmd_vel, /imu) while guaranteeing
    1:1 correspondence between command segments and simulated steps.
    """
    def __init__(self, label_queue: queue.Queue, phases: list):
        super().__init__('demo_sequence_publisher')
        self._pub    = self.create_publisher(Twist, '/cmd_vel', 10)
        self._lq     = label_queue
        self._phases = phases
        self._lock   = threading.Lock()   # protects self._heading (written by /imu callback thread)

        # heading state — updated from /imu when available
        self._heading        = 0.0
        self._heading_from_imu = False
        self.create_subscription(Imu, '/imu', self._imu_cb, 10)

        self._phase_idx      = 0
        self._step_in_phase  = 0
        self._current_vx     = 0.0
        self._current_vy     = 0.0
        self._done           = False

        self.get_logger().info(
            'DemoSequenceNode ready (driven by simulation steps, not wall-clock).')

    def _imu_cb(self, msg: Imu):
        """Update heading from actual robot orientation — closes the outer loop."""
        q = msg.orientation
        with self._lock:
            self._heading = quat_to_yaw(q.w, q.x, q.y, q.z)
        if not self._heading_from_imu:
            self._heading_from_imu = True
            self.get_logger().info('/imu received — heading loop CLOSED.')

    def step(self):
        """Advance one control step. Called from the Genesis simulation loop
        (main thread) — one call ≙ one simulated 0.02 s control step."""
        if self._done:
            return

        total_steps, vx_tgt, vy_tgt, heading_tgt, label = self._phases[self._phase_idx]

        self._current_vx += float(
            np.clip(vx_tgt - self._current_vx, -VX_RAMP_RATE, VX_RAMP_RATE))
        self._current_vy += float(
            np.clip(vy_tgt - self._current_vy, -VX_RAMP_RATE, VX_RAMP_RATE))

        with self._lock:
            heading = self._heading

        err     = heading_tgt - heading
        err     = ((err + math.pi) % (2 * math.pi)) - math.pi
        yaw_cmd = float(np.clip(YAW_GAIN * err, -1., 1.))
        if math.sqrt(self._current_vx**2 + self._current_vy**2) < 0.1:
            yaw_cmd = 0.0

        # open-loop fallback: only integrate when /imu is unavailable
        if not self._heading_from_imu:
            heading = ((heading + yaw_cmd * CONTROL_DT + math.pi)
                       % (2 * math.pi)) - math.pi
            with self._lock:
                self._heading = heading

        msg = Twist()
        msg.linear.x  = self._current_vx
        msg.linear.y  = self._current_vy
        msg.angular.z = yaw_cmd
        self._pub.publish(msg)

        try:
            self._lq.put_nowait((label, self._current_vx,
                                 heading, heading_tgt))
        except queue.Full:
            try: self._lq.get_nowait()
            except queue.Empty: pass
            self._lq.put_nowait((label, self._current_vx,
                                 heading, heading_tgt))

        self._step_in_phase += 1
        if self._step_in_phase >= total_steps:
            self._phase_idx    += 1
            self._step_in_phase = 0
            if self._phase_idx >= len(self._phases):
                self._done = True
                self._pub.publish(Twist())
            else:
                next_label = self._phases[self._phase_idx][4]
                self.get_logger().info(f'→ Segment {self._phase_idx}: {next_label}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='checkpoints/checkpoints_v4_r48s_acc5x/h1_walk_best.pt')
    parser.add_argument('--out', default='videos/ros2_path_closed_acc5x.mp4')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    device  = 'cpu' if args.cpu else 'cuda'
    backend = gs.cpu if args.cpu else gs.cuda

    phases      = generate_fixed_route()
    total_steps = sum(p[0] for p in phases)
    print(f'Fixed route ({len(phases)} phases, {total_steps} steps total):')
    for i, (steps, vx, vy, hd, label) in enumerate(phases):
        print(f'  [{i}] {label:<12s} {steps:4d} steps  '
              f'vx={vx:.2f}  heading target {math.degrees(hd):+.0f}°')

    gs.init(backend=backend)

    # ── policy ───────────────────────────────────────────────────────────────
    ac = ActorCriticRecurrent(NUM_OBS, NUM_ACTIONS,
                              num_privileged_obs=NUM_OBS + 3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    ac.load_state_dict(ckpt['model'])
    ac.eval()
    print(f'Loaded: {args.checkpoint}  (step {ckpt.get("total_steps","?")})')

    obs_norm = None
    ckpt_dir = os.path.dirname(args.checkpoint) or '.'
    for cand in [os.path.join(ckpt_dir, 'obs_norm_best.pt'),
                 os.path.join(ckpt_dir, 'obs_norm.pt')]:
        if os.path.exists(cand):
            nd = torch.load(cand, map_location=device, weights_only=True)
            if 'obs' in nd:
                obs_norm = {'mean': nd['obs']['mean'].to(device),
                            'var':  nd['obs']['var'].to(device)}
                print(f'Loaded obs normalizer: {cand}')
            break

    # ── scene ────────────────────────────────────────────────────────────────
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, substeps=1),
        show_viewer=False, show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file=URDF_PATH, pos=(0., 0., 1.05)))
    cam = scene.add_camera(pos=(3., -3., 3.5), lookat=(0., 0., 0.9),
                           res=(1280, 720), fov=55)
    scene.build(n_envs=1)

    motor_dofs = torch.tensor(MOTOR_DOFS, device=device)
    robot.set_dofs_kp(KP.to(device), motor_dofs)
    robot.set_dofs_kv(KD.to(device), motor_dofs)
    robot.set_pos(torch.tensor([[0., 0., 1.05]], device=device))
    robot.set_quat(torch.tensor([[1., 0., 0., 0.]], device=device))
    robot.set_dofs_position(DEFAULT_DOF.unsqueeze(0).to(device), motor_dofs)
    robot.zero_all_dofs_velocity()

    commands    = torch.zeros(1, 3, device=device)
    last_action = torch.zeros(1, NUM_ACTIONS, device=device)
    phase       = torch.zeros(1, device=device)
    hidden      = ac.init_hidden(1, device)

    # ── ROS2 setup ───────────────────────────────────────────────────────────
    rclpy.init()

    # shared IMU state: Genesis main thread writes, ImuPublisherNode reads
    imu_state  = {'qw': 1., 'qx': 0., 'qy': 0., 'qz': 0.}

    cmd_queue   = queue.Queue(maxsize=2)
    label_queue = queue.Queue(maxsize=2)

    imu_pub_node  = ImuPublisherNode(imu_state)
    cmdvel_node   = CmdVelNode(cmd_queue)
    demo_node     = DemoSequenceNode(label_queue, phases)

    executor = rclpy.executors.MultiThreadedExecutor()
    for n in [imu_pub_node, cmdvel_node, demo_node]:
        executor.add_node(n)

    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    print(f'ROS2 running.  Closed-loop via /imu → /cmd_vel  ({total_steps} steps)')

    # ── simulation loop ───────────────────────────────────────────────────────
    import imageio
    frames = []
    path_positions = []
    current_label   = 'Waiting...'
    current_vx_disp = 0.0
    current_heading = 0.0
    current_tgt     = 0.0

    for step in range(total_steps):
        # ── 1. drive the demo sequencer for exactly one simulated control step ──
        # (sim-time driven, not wall-clock — see DemoSequenceNode docstring for why)
        demo_node.step()

        # block briefly for the /cmd_vel round-trip (publish → DDS → subscriber
        # callback → queue). On localhost this resolves in << 1 ms.
        try:
            vx, vy, yaw_rate = cmd_queue.get(timeout=1.0)
            commands[0, 0] = vx
            commands[0, 1] = vy
            commands[0, 2] = yaw_rate
        except queue.Empty:
            pass  # hold previous command (should not happen on localhost)

        try:
            current_label, current_vx_disp, current_heading, current_tgt = \
                label_queue.get_nowait()
        except queue.Empty:
            pass

        # ── 2. policy inference ───────────────────────────────────────────────
        obs = get_obs(robot, last_action, commands, phase, device, obs_norm)
        phase = (phase + CONTROL_DT / PHASE_PERIOD) % 1.0

        with torch.no_grad():
            action, hidden = ac.act_deterministic(obs, hidden)
        action = torch.clamp(action, -1., 1.)
        target = DEFAULT_DOF.to(device) + action * 0.25

        for _ in range(4):
            robot.control_dofs_position(target, motor_dofs)
            scene.step()

        last_action = action.clone()

        # ── 3. publish robot orientation to /imu (via shared dict) ───────────
        quat = robot.get_quat()[0].cpu()
        imu_state['qw'] = float(quat[0])
        imu_state['qx'] = float(quat[1])
        imu_state['qy'] = float(quat[2])
        imu_state['qz'] = float(quat[3])

        # ── 4. record & render ────────────────────────────────────────────────
        rpos = robot.get_pos()[0].cpu().numpy()
        path_positions.append((rpos[0], rpos[1]))

        cam.set_pose(
            pos    = (rpos[0] + 3., rpos[1] - 3., 3.5),
            lookat = (rpos[0],      rpos[1],       0.9),
        )
        rgb, _, _, _ = cam.render(rgb=True, depth=False,
                                  segmentation=False, normal=False)
        frames.append(annotate(rgb, current_label, current_vx_disp,
                                current_heading, current_tgt, step))

        if (step + 1) % 100 == 0:
            actual_yaw = quat_to_yaw(float(quat[0]), float(quat[1]),
                                     float(quat[2]), float(quat[3]))
            print(f'  step {step+1:4d}/{total_steps}'
                  f'  pos=({rpos[0]:.2f},{rpos[1]:.2f})'
                  f'  cmd=[vx={float(commands[0,0]):.2f}'
                  f' ω={float(commands[0,2]):+.2f}]'
                  f'  yaw={math.degrees(actual_yaw):+.1f}°'
                  f'  phase={current_label}')

    # ── save ─────────────────────────────────────────────────────────────────
    imageio.mimsave(args.out, frames, fps=50)
    print(f'\nVideo saved → {args.out}')

    path_arr = np.array(path_positions)
    path_npy = args.out.replace('.mp4', '_path.npy')
    np.save(path_npy, path_arr)
    print(f'Path data  → {path_npy}')
    print(f'Final pos: ({path_arr[-1,0]:.2f}, {path_arr[-1,1]:.2f})')

    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
