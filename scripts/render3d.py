"""3D video of trained CQ-learning robots in the warehouse, rendered headless with PyBullet.

PyBullet is used as a kinematic 3D viewer (DIRECT mode + its CPU renderer, no display needed):
the motion comes from the grid policy, interpolated between cells. Frames are piped to ffmpeg.

    .venv/bin/python -m scripts.render3d outputs/warehouse_p0.1 --seed 0 --p-fail 0.1 --replay-seed 1
    .venv/bin/python -m scripts.render3d outputs/warehouse_p0 --seed 0 --orbit

Green halo under a robot = it is in a dangerous joint state and acts on its joint table Q^j.
Red halo = a collision this step. Coloured floor tile with a post = that robot's goal.
"""
import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont

from marl.arena import MultiRobotEnv
from marl.grid import Grid
from marl.replay import load_run, rollout

ROBOT_RGB = [(0.16, 0.47, 0.84), (0.92, 0.41, 0.20)]
ROBOT_HEX = [(42, 120, 214), (235, 104, 52)]
HALO_OK, HALO_BAD = (0.11, 0.69, 0.48, 0.55), (0.89, 0.29, 0.28, 0.7)
YAW = {0: 0.0, 1: math.pi / 2, 2: math.pi, 3: -math.pi / 2}   # right, up, left, down


def font(size):
    for name in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def box(center, half, rgba, client):
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=client)
    return p.createMultiBody(0, -1, v, basePosition=center, physicsClientId=client)


def cylinder(center, radius, length, rgba, client):
    v = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=rgba, physicsClientId=client)
    return p.createMultiBody(0, -1, v, basePosition=center, physicsClientId=client)


class Robot:
    """Body + lidar puck + front marker, moved together."""

    def __init__(self, k, client):
        c = ROBOT_RGB[k]
        self.client = client
        self.parts = [  # (body id, local offset)
            (box((0, 0, 0), (0.30, 0.27, 0.13), (*c, 1), client), (0, 0, 0.17)),
            (box((0, 0, 0), (0.31, 0.28, 0.025), (0.15, 0.15, 0.16, 1), client), (0, 0, 0.035)),
            (cylinder((0, 0, 0), 0.10, 0.08, (0.92, 0.92, 0.9, 1), client), (0, 0, 0.34)),
            (box((0, 0, 0), (0.03, 0.17, 0.05), (0.98, 0.98, 0.95, 1), client), (0.29, 0, 0.2)),
        ]
        self.halo = cylinder((0, 0, -5), 0.47, 0.012, HALO_OK, client)
        self.bad = cylinder((0, 0, -5), 0.52, 0.010, HALO_BAD, client)

    def place(self, x, y, yaw, coord, collided, visible=True):
        q = p.getQuaternionFromEuler((0, 0, yaw))
        cy, sy = math.cos(yaw), math.sin(yaw)
        for body, (ox, oy, oz) in self.parts:
            pos = (x + cy * ox - sy * oy, y + sy * ox + cy * oy, oz if visible else -5)
            p.resetBasePositionAndOrientation(body, pos, q, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.halo, (x, y, 0.012 if coord and visible else -5), (0, 0, 0, 1),
                                          physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.bad, (x, y, 0.008 if collided and visible else -5), (0, 0, 0, 1),
                                          physicsClientId=self.client)


def build_scene(grid, goals, client):
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    w, h = grid.width, grid.height
    box((w / 2, h / 2, -0.01), (w / 2 + 0.6, h / 2 + 0.6, 0.01), (0.86, 0.86, 0.84, 1), client)   # floor
    for x in range(w + 1):   # floor grid lines
        box((x, h / 2, 0.001), (0.008, h / 2, 0.001), (0.74, 0.74, 0.72, 1), client)
    for y in range(h + 1):
        box((w / 2, y, 0.001), (w / 2, 0.008, 0.001), (0.74, 0.74, 0.72, 1), client)
    free = set(grid.free)
    rng = np.random.default_rng(3)
    goods = [(0.80, 0.62, 0.38, 1), (0.70, 0.52, 0.30, 1), (0.85, 0.72, 0.50, 1)]
    for y in range(h):
        for x in range(w):
            if (x, y) in free:
                continue
            cx, cy = x + 0.5, y + 0.5   # shelf rack: frame + three levels of boxes
            for zx, zy in ((-0.44, -0.44), (-0.44, 0.44), (0.44, -0.44), (0.44, 0.44)):
                box((cx + zx, cy + zy, 0.75), (0.03, 0.03, 0.75), (0.25, 0.33, 0.45, 1), client)
            for level in (0.05, 0.55, 1.05):
                box((cx, cy, level), (0.47, 0.47, 0.025), (0.36, 0.44, 0.55, 1), client)
                for bx in (-0.22, 0.22):
                    if rng.random() < 0.85:
                        s = rng.uniform(0.15, 0.2)
                        box((cx + bx, cy + rng.uniform(-0.15, 0.15), level + 0.025 + s), (s, s, s),
                            goods[int(rng.integers(3))], client)
    wall = (0.62, 0.65, 0.70, 1)
    box((w / 2, -0.06, 0.4), (w / 2 + 0.12, 0.06, 0.4), wall, client)
    box((w / 2, h + 0.06, 0.4), (w / 2 + 0.12, 0.06, 0.4), wall, client)
    box((-0.06, h / 2, 0.4), (0.06, h / 2, 0.4), wall, client)
    box((w + 0.06, h / 2, 0.4), (0.06, h / 2, 0.4), wall, client)
    for k, g in enumerate(goals):   # goal tile + post with a coloured cap
        x, y = grid.cell(int(g))
        c = ROBOT_RGB[k]
        box((x + 0.5, y + 0.5, 0.006), (0.45, 0.45, 0.006), (*c, 0.45), client)
        cylinder((x + 0.85, y + 0.85, 0.45), 0.025, 0.9, (0.3, 0.3, 0.3, 1), client)
        box((x + 0.85, y + 0.85, 0.95), (0.09, 0.09, 0.09), (*c, 1), client)


def camera(grid, t_frac, orbit, width, height):
    w, h = grid.width, grid.height
    target = (w / 2, h / 2 - 0.3, 0)
    yaw = 15 * math.sin(2 * math.pi * t_frac) if orbit else 0
    view = p.computeViewMatrixFromYawPitchRoll(target, distance=15.8, yaw=yaw, pitch=-55, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=48, aspect=width / height, nearVal=0.1, farVal=60)
    return view, proj


def overlay(img, t, frame, n, steps_done, title, sub):
    d = ImageDraw.Draw(img)
    W, H = img.size
    d.rectangle([0, 0, W, 74], fill=(252, 252, 251))
    d.text((24, 12), title, fill=(11, 11, 11), font=font(26))
    d.text((24, 46), sub, fill=(82, 81, 78), font=font(17))
    d.rectangle([0, H - 58, W, H], fill=(252, 252, 251))
    x = 24
    d.text((x, H - 42), f't = {t}', fill=(11, 11, 11), font=font(20))
    x += 110
    for k in range(n):
        if steps_done[k] is not None:
            status = f'arrived at t = {steps_done[k]}'
        elif frame['coord'][k]:
            status = 'coordinating (joint table)'
        else:
            status = 'own single-robot policy'
        d.ellipse([x, H - 38, x + 18, H - 20], fill=ROBOT_HEX[k])
        d.text((x + 26, H - 42), f'Robot {k + 1}: {status}', fill=(11, 11, 11), font=font(19))
        x += 430
    d.ellipse([W - 270, 14, W - 252, 32], fill=(27, 175, 122))
    d.text((W - 244, 13), 'green halo = coordinating', fill=(11, 11, 11), font=font(16))
    d.rectangle([W - 270, 44, W - 252, 60], fill=(140, 140, 140))
    d.text((W - 244, 42), 'tile + post = goal', fill=(11, 11, 11), font=font(16))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--map', default='warehouse')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--method', default='cq')
    ap.add_argument('--p-fail', type=float, default=0.0)
    ap.add_argument('--replay-seed', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=80)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--sub', type=int, default=10, help='video frames per grid step')
    ap.add_argument('--size', default='1280x720')
    ap.add_argument('--orbit', action='store_true', help='slowly swing the camera')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    width, height = map(int, args.size.split('x'))
    grid = Grid.load(args.map)
    policies, meta = load_run(Path(args.run) / args.map / f'{args.method}_seed{args.seed}.npz')
    env = MultiRobotEnv(grid, p_fail=args.p_fail, seed=args.replay_seed)
    frames = rollout(env, policies, max_steps=args.max_steps)
    n = env.n
    arrive = [next((f['t'] for f in frames if f['done'][k]), None) for k in range(n)]
    print(f'{args.method} seed {args.seed}: steps to goal {arrive}, '
          f'coordinating steps {[sum(f["coord"][k] for f in frames) for k in range(n)]}')

    client = p.connect(p.DIRECT)
    build_scene(grid, env.goals, client)
    robots = [Robot(k, client) for k in range(n)]
    out = Path(args.out or Path(args.run) / 'figures' / f'cq3d_{args.map}_seed{args.seed}.mp4')
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{width}x{height}', '-r', str(args.fps), '-i', '-', '-c:v', 'libx264',
                           '-pix_fmt', 'yuv420p', '-crf', '20', str(out)], stdin=subprocess.PIPE)
    title = 'CQ-learning: two robots in the warehouse'
    sub = (f'Own single-robot Q-table everywhere, joint table only in learned danger states'
           f'  |  failed moves {args.p_fail:.0%}  |  seed {args.seed}')
    yaw = [None] * n
    total = (len(frames) + 2) * args.sub
    still_at, still = None, None
    k_frame = 0
    for i in range(len(frames) + 2):
        cur = frames[min(i, len(frames) - 1)]
        nxt = frames[min(i + 1, len(frames) - 1)]
        collided = {c[0] for c in nxt['conflicts']} | {c[1] for c in nxt['conflicts'] if c[2] != 'occupied'}
        for s in range(args.sub):
            a = s / args.sub
            ease = a * a * (3 - 2 * a)
            for k in range(n):
                visible = not cur['done'][k]
                x0, y0 = grid.cell(int(cur['pos'][k]))
                # A robot that reaches its goal leaves the grid; animate it onto the goal tile first.
                x1, y1 = grid.cell(int(env.goals[k] if nxt['done'][k] else nxt['pos'][k]))
                moving = (x1, y1) != (x0, y0)
                if moving:
                    target = math.atan2(y1 - y0, x1 - x0)
                    yaw[k] = target if yaw[k] is None else yaw[k]
                    yaw[k] += ((target - yaw[k] + math.pi) % (2 * math.pi) - math.pi) * 0.45
                elif yaw[k] is None:
                    yaw[k] = 0.0
                # Blocked move (collision or wall): a small nudge toward the intended cell and back.
                act = nxt['actions'][k] if nxt['actions'] is not None and i + 1 < len(frames) else None
                if not moving and visible and act is not None and \
                        (k in collided or grid.bumps[int(cur['pos'][k]), act]):
                    dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][act]
                    nudge = 0.12 * math.sin(math.pi * a)
                    x0, y0 = x0 + dx * nudge, y0 + dy * nudge
                x = x0 + (x1 - x0) * ease + 0.5 if moving else x0 + 0.5
                y = y0 + (y1 - y0) * ease + 0.5 if moving else y0 + 0.5
                robots[k].place(x, y, yaw[k], cur['coord'][k], k in collided and s < args.sub * 0.7,
                                visible=visible)
            view, proj = camera(grid, k_frame / max(1, total), args.orbit, width, height)
            _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj, shadow=1,
                                                lightDirection=(-4, -6, 10), lightAmbientCoeff=0.55,
                                                lightDiffuseCoeff=0.6, lightSpecularCoeff=0.08,
                                                renderer=p.ER_TINY_RENDERER, physicsClientId=client)
            img = Image.fromarray(np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3])
            done_so_far = [arrive[k] if arrive[k] is not None and arrive[k] <= cur['t'] else None for k in range(n)]
            img = overlay(img, cur['t'], cur, n, done_so_far, title, sub)
            if still_at is None and any(cur['coord']) and s == args.sub // 2:
                still_at, still = cur['t'], img.copy()
            ff.stdin.write(img.tobytes())
            k_frame += 1
        print(f'\r  rendered step {i}/{len(frames) + 1}', end='', flush=True)
    for _ in range(args.fps * 2):   # hold the final frame
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    p.disconnect(client)
    (still or img).save(out.with_suffix('.png'))
    print(f'\nwrote {out} and still {out.with_suffix(".png")}' + (f' (t = {still_at})' if still_at else ''))


if __name__ == '__main__':
    main()
