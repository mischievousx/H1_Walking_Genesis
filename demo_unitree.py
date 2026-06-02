"""
Demo using the Unitree pretrained H1 walking policy (TorchScript).

DOF order mismatch between Unitree (MuJoCo) and Genesis:
  Unitree: L_yaw L_roll L_pitch L_knee L_ankle | R_yaw R_roll R_pitch R_knee R_ankle
  Genesis: L_yaw R_yaw L_roll R_roll L_pitch R_pitch L_knee R_knee L_ankle R_ankle

Usage:
    python demo_unitree.py
    python demo_unitree.py --out videos/demo_unitree.mp4
"""
import os
import argparse
import math
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

URDF_PATH    = os.path.join(os.path.dirname(__file__), 'robot/urdf/h1.urdf')
POLICY_PATH  = '/root/autodl-tmp/unitree_rl_gym/deploy/pre_train/h1/motion.pt'
NUM_OBS      = 41
NUM_ACTIONS  = 10
MOTOR_DOFS   = list(range(6, 16))

# Genesis DOF order (L/R interleaved per joint type)
DEFAULT_DOF_GENESIS = torch.tensor([0., 0., 0., 0., -0.1, -0.1, 0.3, 0.3, -0.2, -0.2])
# Unitree DOF order (full left leg then full right leg)
DEFAULT_DOF_UNITREE = torch.tensor([0., 0., -0.1, 0.3, -0.2, 0., 0., -0.1, 0.3, -0.2])

KP = torch.tensor([150, 150, 150, 150, 150, 150, 200, 200, 40, 40], dtype=torch.float32)
KD = torch.tensor([  2,   2,   2,   2,   2,   2,   4,   4,  2,  2], dtype=torch.float32)

# Reindex: Genesis order → Unitree order  (for obs: dof_pos, dof_vel, last_action)
G2U = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
# Reindex: Unitree order → Genesis order  (for action → position target)
U2G = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]

VX_RAMP_RATE = 0.02   # m/s per control step

PHASES = [
    (100, 0.5, 0.0,  0.0,         "Forward"),
    (300, 0.4, 0.0,  math.pi/2,   "Left turn"),
    (200, 0.5, 0.0,  math.pi/2,   "Forward"),
    (300, 0.4, 0.0,  0.0,         "Right turn"),
    (150, 0.5, 0.0,  0.0,         "Forward"),
    (100, 0.0, 0.0,  0.0,         "Stop"),
]


def quat_rotate_inverse(q, v):
    qw = q[:, 0]; qv = q[:, 1:]
    a = v * (2. * qw**2 - 1.).unsqueeze(-1)
    b = torch.cross(qv, v, dim=1) * (2. * qw).unsqueeze(-1)
    c = qv * torch.sum(qv * v, dim=1, keepdim=True) * 2.
    return a - b + c


def get_obs(robot, last_action_unitree, commands, phase_val, device):
    """Build 41-dim obs in Unitree order."""
    quat    = robot.get_quat()
    ang_vel = quat_rotate_inverse(quat, robot.get_ang())
    proj_g  = quat_rotate_inverse(quat, torch.tensor([[0., 0., -1.]], device=device))

    motor_dofs = torch.tensor(MOTOR_DOFS, device=device)
    # Get in Genesis order, reindex to Unitree order, subtract Unitree defaults
    dof_pos_g = robot.get_dofs_position(motor_dofs)
    dof_pos   = dof_pos_g[:, G2U] - DEFAULT_DOF_UNITREE.to(device)
    dof_vel   = robot.get_dofs_velocity(motor_dofs)[:, G2U]

    cmd_scale = torch.tensor([2., 2., 0.25], device=device)
    sin_phase = torch.sin(2 * math.pi * phase_val).unsqueeze(1)
    cos_phase = torch.cos(2 * math.pi * phase_val).unsqueeze(1)

    obs = torch.cat([
        ang_vel * 0.25,
        proj_g,
        commands * cmd_scale,
        dof_pos,
        dof_vel * 0.05,
        last_action_unitree,
        sin_phase,
        cos_phase,
    ], dim=-1)
    return torch.clamp(obs, -5., 5.)


def _annotate(rgb: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    text = f"Phase: {label}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = 14
    x = img.width - tw - margin * 2
    y = margin
    draw.rectangle([x - margin, y - margin, x + tw + margin, y + th + margin], fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default=POLICY_PATH)
    parser.add_argument('--out',    default='videos/demo_unitree.mp4')
    parser.add_argument('--cpu',    action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    device  = 'cpu' if args.cpu else 'cuda'
    backend = gs.cpu if args.cpu else gs.cuda

    gs.init(backend=backend)

    policy = torch.jit.load(args.policy, map_location=device)
    policy.eval()
    print(f"Loaded Unitree policy: {args.policy}")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, substeps=1),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file=URDF_PATH, pos=(0., 0., 1.05)))
    cam = scene.add_camera(
        pos=(3., -3., 3.5),
        lookat=(0., 0., 0.9),
        res=(1280, 720),
        fov=55,
    )
    scene.build(n_envs=1)

    motor_dofs = torch.tensor(MOTOR_DOFS, device=device)
    robot.set_dofs_kp(KP.to(device), motor_dofs)
    robot.set_dofs_kv(KD.to(device), motor_dofs)

    robot.set_pos(torch.tensor([[0., 0., 1.05]], device=device))
    robot.set_quat(torch.tensor([[1., 0., 0., 0.]], device=device))
    robot.set_dofs_position(DEFAULT_DOF_GENESIS.unsqueeze(0).to(device), motor_dofs)
    robot.zero_all_dofs_velocity()

    commands           = torch.tensor([[0., 0., 0.]], device=device)
    last_action_unitree = torch.zeros(1, NUM_ACTIONS, device=device)
    phase              = torch.zeros(1, device=device)

    import imageio
    frames = []
    total_steps = sum(p[0] for p in PHASES)
    print(f"Total steps: {total_steps} (~{total_steps/50:.1f}s)  →  {args.out}")

    current_vx = 0.0
    current_vy = 0.0

    for phase_steps, vx, vy, heading_target, label in PHASES:
        print(f"\n[Phase] {label}  (vx={vx}, heading={math.degrees(heading_target):.0f}°, {phase_steps} steps)")
        heading_target_t = torch.tensor([heading_target], device=device)

        for s in range(phase_steps):
            # Linear ramp
            current_vx += float(np.clip(vx - current_vx, -VX_RAMP_RATE, VX_RAMP_RATE))
            current_vy += float(np.clip(vy - current_vy, -VX_RAMP_RATE, VX_RAMP_RATE))
            commands[0, 0] = current_vx
            commands[0, 1] = current_vy

            # Heading → yaw command
            quat = robot.get_quat()
            qw, qx, qy, qz = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
            heading = torch.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
            err = heading_target_t - heading
            err = ((err + math.pi) % (2 * math.pi)) - math.pi
            yaw_cmd = torch.clamp(0.5 * err, -1., 1.)
            if torch.norm(commands[0, :2]) < 0.1:
                yaw_cmd = yaw_cmd * 0.
            commands[0, 2] = yaw_cmd

            obs = get_obs(robot, last_action_unitree, commands, phase, device)
            phase = (phase + 0.02 / 0.8) % 1.0

            with torch.no_grad():
                action_unitree = policy(obs)   # TorchScript, hidden managed internally
            action_unitree = torch.clamp(action_unitree, -1., 1.)

            # Reindex action Unitree → Genesis, apply to robot
            action_genesis = action_unitree[:, U2G]
            target = DEFAULT_DOF_GENESIS.to(device) + action_genesis * 0.25

            for _ in range(4):
                robot.control_dofs_position(target, motor_dofs)
                scene.step()

            last_action_unitree = action_unitree.clone()

            rpos = robot.get_pos()[0].cpu().numpy()
            cam.set_pose(
                pos    = (rpos[0] + 3., rpos[1] - 3., 3.5),
                lookat = (rpos[0],      rpos[1],       0.9),
            )
            rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
            frames.append(_annotate(rgb, f"{label}  vx={current_vx:.2f}"))

        print(f"  Done. Robot pos: {robot.get_pos()[0].cpu().numpy()}")

    imageio.mimsave(args.out, frames, fps=50)
    print(f"\nVideo saved → {args.out}")


if __name__ == '__main__':
    main()
