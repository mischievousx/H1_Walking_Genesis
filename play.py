"""
Visualize a trained H1 policy by rendering a video.

Usage:
    python play.py checkpoints/h1_walk_000100.pt
    python play.py checkpoints/h1_walk_final.pt --steps 500 --out videos/walk.mp4
"""
import os
import sys
import argparse
import torch

# Limit quadrants GPU memory pre-allocation so play can run alongside training
import quadrants as _qd
_qd_init_orig = _qd.init
def _qd_init_patched(**kwargs):
    kwargs.setdefault('device_memory_GB', 8.0)
    return _qd_init_orig(**kwargs)
_qd.init = _qd_init_patched

import genesis as gs

from ppo import ActorCriticRecurrent

URDF_PATH    = os.path.join(os.path.dirname(__file__), 'robot/urdf/h1.urdf')
NUM_OBS      = 41
NUM_ACTIONS  = 10
DEFAULT_DOF  = torch.tensor([0.,0.,0.,0.,-0.1,-0.1,0.3,0.3,-0.2,-0.2])
MOTOR_DOFS   = list(range(6, 16))
KP = torch.tensor([150,150,150,150,150,150,200,200,40,40], dtype=torch.float32)
KD = torch.tensor([  2,  2,  2,  2,  2,  2,  4,  4, 2, 2], dtype=torch.float32)


def quat_rotate_inverse(q, v):
    qw = q[:, 0]; qv = q[:, 1:]
    a = v * (2. * qw**2 - 1.).unsqueeze(-1)
    b = torch.cross(qv, v, dim=1) * (2. * qw).unsqueeze(-1)
    c = qv * torch.sum(qv * v, dim=1, keepdim=True) * 2.
    return a - b + c


def get_obs(robot, last_action, commands, phase, device, obs_norm=None):
    quat    = robot.get_quat()
    ang_vel = quat_rotate_inverse(quat, robot.get_ang())
    proj_g  = quat_rotate_inverse(
        quat, torch.tensor([[0., 0., -1.]], device=device)
    )
    motor_dofs = torch.tensor(MOTOR_DOFS, device=device)
    dof_pos = robot.get_dofs_position(motor_dofs) - DEFAULT_DOF.to(device)
    dof_vel = robot.get_dofs_velocity(motor_dofs)
    cmd_scale = torch.tensor([2., 2., 0.25], device=device)
    sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
    cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)
    obs = torch.cat([
        ang_vel * 0.25,         # 3
        proj_g,                 # 3
        commands * cmd_scale,   # 3
        dof_pos,                # 10
        dof_vel * 0.05,         # 10
        last_action,            # 10
        sin_phase,              # 1
        cos_phase,              # 1
    ], dim=-1)                  # = 41
    obs = torch.clamp(obs, -5., 5.)
    if obs_norm is not None:
        obs = torch.clamp((obs - obs_norm['mean']) / torch.sqrt(obs_norm['var'] + 1e-4), -10., 10.)
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint',             help='Path to .pt checkpoint')
    parser.add_argument('--steps', type=int, default=600,  help='Simulation steps to record')
    parser.add_argument('--out',   default='videos/play.mp4', help='Output video path')
    parser.add_argument('--vx',    type=float, default=0.5,   help='Commanded forward speed (m/s)')
    parser.add_argument('--vy',    type=float, default=0.0,   help='Commanded lateral speed (m/s)')
    parser.add_argument('--yaw',   type=float, default=0.0,   help='Commanded yaw rate (rad/s)')
    parser.add_argument('--cpu',   action='store_true',       help='Use CPU backend')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    device  = 'cpu' if args.cpu else 'cuda'
    backend = gs.cpu if args.cpu else gs.cuda

    # ── Init genesis first (must precede any PyTorch CUDA ops) ───────────────
    gs.init(backend=backend)

    # ── Load policy ──────────────────────────────────────────────────────────
    ac = ActorCriticRecurrent(NUM_OBS, NUM_ACTIONS, num_privileged_obs=NUM_OBS+3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    ac.load_state_dict(ckpt['model'])
    ac.eval()
    print(f"Loaded checkpoint: {args.checkpoint}  "
          f"(step {ckpt.get('total_steps', '?')})")

    # ── Load obs normalizer if present (v5 checkpoints) ──────────────────────
    obs_norm = None
    ckpt_dir = os.path.dirname(args.checkpoint)
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

    # ── Build scene ──────────────────────────────────────────────────────────
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, substeps=1),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, pos=(0., 0., 1.05))
    )

    cam = scene.add_camera(
        pos=(0., -3.0, 1.2),
        lookat=(0., 0., 0.9),
        res=(1280, 720),
        fov=50,
    )

    scene.build(n_envs=1)

    motor_dofs = torch.tensor(MOTOR_DOFS, device=device)
    robot.set_dofs_kp(KP.to(device), motor_dofs)
    robot.set_dofs_kv(KD.to(device), motor_dofs)

    # ── Reset robot ──────────────────────────────────────────────────────────
    pos  = torch.tensor([[0., 0., 1.05]], device=device)
    quat = torch.tensor([[1., 0., 0., 0.]], device=device)
    robot.set_pos(pos)
    robot.set_quat(quat)
    robot.set_dofs_position(DEFAULT_DOF.unsqueeze(0).to(device), motor_dofs)
    robot.zero_all_dofs_velocity()

    commands        = torch.tensor([[args.vx, args.vy, 0.]], device=device)
    last_action     = torch.zeros(1, NUM_ACTIONS, device=device)
    phase           = torch.zeros(1, device=device)
    # heading target: 0 = forward (+x), matches training convention
    heading_target  = torch.tensor([args.yaw], device=device)

    # LSTM hidden state (1 env)
    hidden = ac.init_hidden(1, device)

    # ── Record ───────────────────────────────────────────────────────────────
    import imageio
    frames = []
    print(f"Recording {args.steps} steps → {args.out}  (vx={args.vx}, vy={args.vy}, heading={args.yaw})")

    for step in range(args.steps):
        # Update yaw command from heading error, matching env._update_heading_command()
        quat    = robot.get_quat()
        qw, qx, qy, qz = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        heading = torch.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        err     = heading_target - heading
        err     = ((err + 3.14159) % (2 * 3.14159)) - 3.14159
        yaw_cmd = torch.clamp(0.5 * err, -1., 1.)
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

        rpos = robot.get_pos()[0].cpu().numpy()
        cam.set_pose(
            pos    = (rpos[0], rpos[1] - 3.0, 1.2),
            lookat = (rpos[0], rpos[1],        0.9),
        )

        rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
        frames.append(rgb)

        if (step + 1) % 100 == 0:
            print(f"  {step+1}/{args.steps} steps done")

    imageio.mimsave(args.out, frames, fps=50)
    print(f"Video saved to {args.out}")


if __name__ == '__main__':
    main()
