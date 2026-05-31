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
import genesis as gs

from ppo import ActorCritic

URDF_PATH    = os.path.join(os.path.dirname(__file__), 'robot/urdf/h1.urdf')
NUM_OBS      = 42
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


def get_obs(robot, last_action, commands, device):
    quat    = robot.get_quat()
    lin_vel = quat_rotate_inverse(quat, robot.get_vel())
    ang_vel = quat_rotate_inverse(quat, robot.get_ang())
    proj_g  = quat_rotate_inverse(
        quat, torch.tensor([[0., 0., -1.]], device=device)
    )
    motor_dofs = torch.tensor(MOTOR_DOFS, device=device)
    dof_pos = robot.get_dofs_position(motor_dofs) - DEFAULT_DOF.to(device)
    dof_vel = robot.get_dofs_velocity(motor_dofs)
    cmd_scale = torch.tensor([2., 2., 0.25], device=device)
    obs = torch.cat([
        lin_vel,
        ang_vel * 0.25,
        proj_g,
        commands * cmd_scale,
        dof_pos,
        dof_vel * 0.05,
        last_action,
    ], dim=-1)
    return torch.clamp(obs, -5., 5.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint',             help='Path to .pt checkpoint')
    parser.add_argument('--steps', type=int, default=600,  help='Simulation steps to record')
    parser.add_argument('--out',   default='videos/play.mp4', help='Output video path')
    parser.add_argument('--vx',    type=float, default=1.0,   help='Commanded forward speed (m/s)')
    parser.add_argument('--cpu',   action='store_true',       help='Use CPU backend')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    device  = 'cpu' if args.cpu else 'cuda'
    backend = gs.cpu if args.cpu else gs.cuda

    # ── Load policy ──────────────────────────────────────────────────────────
    ac = ActorCritic(NUM_OBS, NUM_ACTIONS).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ac.load_state_dict(ckpt['model'])
    ac.eval()
    print(f"Loaded checkpoint: {args.checkpoint}  "
          f"(step {ckpt.get('total_steps', '?')})")

    # ── Build scene ──────────────────────────────────────────────────────────
    gs.init(backend=backend)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, substeps=1),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, pos=(0., 0., 1.05))
    )

    # Side-follow camera (updated manually each frame)
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

    commands    = torch.tensor([[args.vx, 0., 0.]], device=device)
    last_action = torch.zeros(1, NUM_ACTIONS, device=device)

    # ── Record ───────────────────────────────────────────────────────────────
    import imageio
    frames = []
    print(f"Recording {args.steps} steps → {args.out}  (vx={args.vx} m/s)")

    for step in range(args.steps):
        obs = get_obs(robot, last_action, commands, device)

        with torch.no_grad():
            action = ac.actor(obs)          # deterministic (mean action)
        action      = torch.clamp(action, -1., 1.)
        target      = DEFAULT_DOF.to(device) + action * 0.35

        for _ in range(4):                  # 4 sim steps per control step
            robot.control_dofs_position(target, motor_dofs)
            scene.step()

        last_action = action.clone()

        # Update camera to follow robot from the side
        rpos = robot.get_pos()[0].cpu().numpy()   # [x, y, z]
        cam.set_pose(
            pos    = (rpos[0],       rpos[1] - 3.0, 1.2),
            lookat = (rpos[0],       rpos[1],       0.9),
        )

        # Render frame
        rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
        frames.append(rgb)

        if (step + 1) % 100 == 0:
            print(f"  {step+1}/{args.steps} steps done")

    imageio.mimsave(args.out, frames, fps=50)
    print(f"Video saved to {args.out}")


if __name__ == '__main__':
    main()
