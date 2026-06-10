"""
Demo Sequence Node
==================
Replicates the PHASES from demo_sequence.py as a ROS2 cmd_vel publisher.
Publishes geometry_msgs/Twist to /cmd_vel at 50 Hz.

Phases (same as demo_sequence.py):
  Forward → Left turn → Forward → Right turn → Forward → Stop

Usage:
  ros2 run ros2_h1_deploy demo_sequence_node
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ── phase definition (mirrors demo_sequence.py PHASES) ──────────────────────
# (steps, vx_target, vy_target, heading_target_rad, label)
PHASES = [
    (200, 0.5, 0.0,  0.0,          'Forward'),
    (300, 0.4, 0.0,  math.pi / 2,  'Left turn'),
    (200, 0.5, 0.0,  math.pi / 2,  'Forward'),
    (300, 0.4, 0.0,  0.0,          'Right turn'),
    (150, 0.5, 0.0,  0.0,          'Forward'),
    (100, 0.0, 0.0,  0.0,          'Stop'),
]

CONTROL_DT   = 0.02     # 50 Hz
VX_RAMP_RATE = 0.02     # m/s per step (1.0 m/s per second)
YAW_GAIN     = 1.0      # proportional gain for heading → yaw_rate command


class DemoSequenceNode(Node):
    """
    Publishes /cmd_vel according to the hardcoded PHASES sequence.
    Heading error is converted to a yaw_rate command (same logic as demo_sequence.py).

    NOTE: This node does NOT read odometry; it estimates the heading from an
    internal integrator (yaw_rate × dt). For real deployment with accurate
    heading feedback, replace `self._heading` with a subscriber to /odom or /imu.
    """

    def __init__(self):
        super().__init__('demo_sequence_node')

        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # internal state
        self._current_vx = 0.0
        self._current_vy = 0.0
        self._heading    = 0.0     # estimated yaw (rad), integrates yaw_rate

        # phase iterator
        self._phases     = list(PHASES)
        self._phase_idx  = 0
        self._step_in_phase = 0
        self._done       = False

        phase = self._phases[0]
        self.get_logger().info(
            f'Demo sequence starting.  Total phases: {len(self._phases)}\n'
            f'  Phase 0: {phase[4]}  (vx={phase[1]}, heading={math.degrees(phase[3]):.0f}°, '
            f'{phase[0]} steps = {phase[0] * CONTROL_DT:.1f}s)')

        self._timer = self.create_timer(CONTROL_DT, self._tick)

    def _tick(self):
        if self._done:
            return

        total_steps, vx_tgt, vy_tgt, heading_tgt, label = self._phases[self._phase_idx]

        # ramp vx/vy toward phase target
        self._current_vx += float(
            np.clip(vx_tgt - self._current_vx, -VX_RAMP_RATE, VX_RAMP_RATE))
        self._current_vy += float(
            np.clip(vy_tgt - self._current_vy, -VX_RAMP_RATE, VX_RAMP_RATE))

        # heading error → yaw_rate command
        err = heading_tgt - self._heading
        err = ((err + math.pi) % (2 * math.pi)) - math.pi   # wrap to [-π, π]
        yaw_rate = float(np.clip(YAW_GAIN * err, -1., 1.))
        if math.sqrt(self._current_vx**2 + self._current_vy**2) < 0.1:
            yaw_rate = 0.0  # no yaw correction when nearly stopped

        # integrate heading estimate
        self._heading += yaw_rate * CONTROL_DT
        self._heading  = ((self._heading + math.pi) % (2 * math.pi)) - math.pi

        # publish
        msg = Twist()
        msg.linear.x  = self._current_vx
        msg.linear.y  = self._current_vy
        msg.angular.z = yaw_rate
        self._pub.publish(msg)

        self._step_in_phase += 1

        # advance phase
        if self._step_in_phase >= total_steps:
            self._phase_idx     += 1
            self._step_in_phase  = 0

            if self._phase_idx >= len(self._phases):
                self.get_logger().info('Demo sequence complete.')
                self._done = True
                # publish zero command to stop
                self._pub.publish(Twist())
                return

            next_phase = self._phases[self._phase_idx]
            self.get_logger().info(
                f'Phase {self._phase_idx}: {next_phase[4]}  '
                f'(vx={next_phase[1]}, heading={math.degrees(next_phase[3]):.0f}°, '
                f'{next_phase[0]} steps)')


def main(args=None):
    rclpy.init(args=args)
    node = DemoSequenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
