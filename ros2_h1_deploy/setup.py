from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros2_h1_deploy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mischievousx',
    maintainer_email='leoj31406@gmail.com',
    description='ROS2 deployment node for H1 humanoid walking RL policy',
    license='MIT',
    entry_points={
        'console_scripts': [
            'h1_rl_node         = ros2_h1_deploy.h1_rl_node:main',
            'demo_sequence_node = ros2_h1_deploy.demo_sequence_node:main',
        ],
    },
)
