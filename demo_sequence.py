"""
Sequence demo using a trained H1 policy.
Phases: forward → left turn → forward → right turn → forward → stop

Usage:
    python demo_sequence.py
    python demo_sequence.py --checkpoint checkpoints_v2/h1_walk_002000.pt --out videos/demo_seq.mp4
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

from ppo import ActorCriticRecurrent

URDF_PATH   = os.path.join(os.path.dirname(__file__), 'robot/urdf/h1.urdf')
NUM_OBS     = 41
NUM_ACTIONS = 10
DEFAULT_DOF = torch.tensor([0., 0., 0., 0., -0.1, -0.1, 0.3, 0.3, -0.2, -0.2])
MOTOR_DOFS  = list(range(6, 16))
KP = torch.tensor([150, 150, 150, 150, 150, 150, 200, 200, 40, 40], dtype=torch.float32)
KD = torch.tensor([  2,   2,   2,   2,   2,   2,   4,   4,  2,  2], dtype=torch.float32)

VX_RAMP_RATE = 0.02   # m/s per control step → 1.0 m/s/s at 50 Hz

# Each phase: (steps, vx, vy, heading_target_rad, label)
# 50 control-steps ≈ 1 second  (control dt = 0.02s)
PHASES = [
    (200, 0.5, 0.0,  0.0,         "Forward"),
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
        ang_vel * 0.25,
        proj_g,
        commands * cmd_scale,
        dof_pos,
        dof_vel * 0.05,
        last_action,
        sin_phase,
        cos_phase,
    ], dim=-1)
    obs = torch.clamp(obs, -5., 5.)
    if obs_norm is not None:
        obs = torch.clamp((obs - obs_norm['mean']) / torch.sqrt(obs_norm['var'] + 1e-4), -10., 10.)
    return obs


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
    draw.rectangle([x - margin, y - margin, x + tw + margin, y + th + margin],
                   fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints_v5_003/h1_walk_best.pt')
    parser.add_argument('--out',        default='videos/demo_sequence.mp4')
    parser.add_argument('--cpu',        action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    device  = 'cpu' if args.cpu else 'cuda'
    backend = gs.cpu if args.cpu else gs.cuda

    # Load policy
    # Init genesis first (must precede any PyTorch CUDA ops)
    gs.init(backend=backend)

    ac = ActorCriticRecurrent(NUM_OBS, NUM_ACTIONS, num_privileged_obs=NUM_OBS+3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    ac.load_state_dict(ckpt['model'])
    ac.eval()
    print(f"Loaded: {args.checkpoint}  (step {ckpt.get('total_steps', '?')})")

    # Load obs normalizer if present (v5 checkpoints)
    obs_norm = None
    ckpt_dir = os.path.dirname(args.checkpoint) or '.'
    for norm_candidate in [
        os.path.join(ckpt_dir, 'obs_norm_best.pt'),
        os.path.join(ckpt_dir, 'obs_norm.pt'),
    ]:
        if os.path.exists(norm_candidate):
            nd = torch.load(norm_candidate, map_location=device, weights_only=True)
            if 'obs' in nd:
                obs_norm = {'mean': nd['obs']['mean'].to(device),
                            'var':  nd['obs']['var'].to(device)}
                print(f"Loaded obs normalizer: {norm_candidate}")
            break
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

    # Reset
    robot.set_pos(torch.tensor([[0., 0., 1.05]], device=device))
    robot.set_quat(torch.tensor([[1., 0., 0., 0.]], device=device))
    robot.set_dofs_position(DEFAULT_DOF.unsqueeze(0).to(device), motor_dofs)
    robot.zero_all_dofs_velocity()

    commands    = torch.tensor([[0., 0., 0.]], device=device)
    last_action = torch.zeros(1, NUM_ACTIONS, device=device)
    phase       = torch.zeros(1, device=device)
    hidden      = ac.init_hidden(1, device)

    import imageio
    frames = []
    total_steps = sum(p[0] for p in PHASES)
    print(f"Total steps: {total_steps} (~{total_steps/50:.1f}s)  →  {args.out}")

    current_vx = 0.0
    current_vy = 0.0
    global_step = 0
    for phase_steps, vx, vy, heading_target, label in PHASES:
        print(f"\n[Phase] {label}  (vx={vx}, heading={math.degrees(heading_target):.0f}°, {phase_steps} steps)")
        heading_target_t = torch.tensor([heading_target], device=device)

        for s in range(phase_steps):
            # Linear ramp vx/vy toward phase target
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
            yaw_cmd = torch.clamp(1.0 * err, -1., 1.)
            if torch.norm(commands[0, :2]) < 0.1:
                yaw_cmd = yaw_cmd * 0.
            commands[0, 2] = yaw_cmd

            obs = get_obs(robot, last_action, commands, phase, device, obs_norm)
            phase = (phase + 0.02 / 0.8) % 1.0

            with torch.no_grad():
                action, hidden = ac.act_deterministic(obs, hidden)
            action = torch.clamp(action, -1., 1.)
            target = DEFAULT_DOF.to(device) + action * 0.25

            for _ in range(4):
                robot.control_dofs_position(target, motor_dofs)
                scene.step()

            last_action = action.clone()

            # Diagonal follow cam
            rpos = robot.get_pos()[0].cpu().numpy()
            cam.set_pose(
                pos    = (rpos[0] + 3., rpos[1] - 3., 3.5),
                lookat = (rpos[0],      rpos[1],       0.9),
            )

            rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
            frames.append(_annotate(rgb, f"{label}  vx={current_vx:.2f}"))
            global_step += 1

        print(f"  Done. Robot pos: {robot.get_pos()[0].cpu().numpy()}")

    imageio.mimsave(args.out, frames, fps=50)
    print(f"\nVideo saved → {args.out}")


if __name__ == '__main__':
    main()
