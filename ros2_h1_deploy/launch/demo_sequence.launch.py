"""
Launch file: demo_sequence
==========================
Starts both the H1 RL controller and the demo sequence command publisher.

Usage:
  ros2 launch ros2_h1_deploy demo_sequence.launch.py \\
      net:=eth0 \\
      checkpoint:=/root/autodl-tmp/h1_genesis_train/checkpoints/checkpoints_v4_r48s_acc5x/h1_walk_best.pt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    net_arg = DeclareLaunchArgument(
        'net', default_value='eth0',
        description='Network interface connected to H1 (e.g. eth0)')

    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint',
        default_value='',
        description='Absolute path to the .pt checkpoint file')

    rl_node = Node(
        package='ros2_h1_deploy',
        executable='h1_rl_node',
        name='h1_rl_node',
        output='screen',
        parameters=[{
            'net':        LaunchConfiguration('net'),
            'checkpoint': LaunchConfiguration('checkpoint'),
        }],
    )

    demo_node = Node(
        package='ros2_h1_deploy',
        executable='demo_sequence_node',
        name='demo_sequence_node',
        output='screen',
    )

    return LaunchDescription([
        net_arg,
        checkpoint_arg,
        rl_node,
        demo_node,
    ])
