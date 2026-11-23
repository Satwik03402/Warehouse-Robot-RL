"""2D replay demo: animate trained greedy policies side by side as a GIF (no display needed).

    python3 -m scripts.replay_gif outputs/warehouse_p0 --seed 0                  # indep | js | cq
    python3 -m scripts.replay_gif outputs/warehouse_p0.1 --seed 3 --methods cq --p-fail 0.1
    python3 -m scripts.replay_gif outputs/table1 --map cit --seed 3

Robots are numbered dots with a fading trail; each robot's goal is a square outline in its colour.
A ring around a CQ robot means it is in a dangerous joint state and acting on its joint table
Q^j; elsewhere it follows its single-robot table. Red flashes mark collisions.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from marl.arena import MultiRobotEnv
from marl.grid import Grid
from marl.replay import load_run, rollout

ROBOT = [(42, 120, 214), (235, 104, 52)]
SURFACE, FREE, SHELF, LINE = (252, 252, 251), (255, 255, 255), (61, 61, 58), (228, 227, 223)
INK, INK2, RING, BAD = (11, 11, 11), (82, 81, 78), (27, 175, 122), (227, 73, 72)
LABELS = {'indep': 'Indep (own cell only)', 'js': 'JS (full joint state)', 'cq': 'CQ-learning'}


def font(size):
    for name in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_panel(grid, frames, t, title, cell, n_states):
    w, h = grid.width * cell, grid.height * cell
    top, bottom = 46, 40
    img = Image.new('RGB', (w + 2, h + top + bottom), SURFACE)
    d = ImageDraw.Draw(img)
    f_title, f_small, f_num = font(15), font(12), font(int(cell * 0.45))

    def xy(c):
        x, y = c
        return 1 + x * cell, top + (grid.height - 1 - y) * cell

    free = set(grid.free)
    for y in range(grid.height):
        for x in range(grid.width):
            x0, y0 = xy((x, y))
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=FREE if (x, y) in free else SHELF, outline=LINE)
    for a, b in grid.walls:
        (x1, y1), (x2, y2) = sorted([tuple(a), tuple(b)])
        if x1 == x2:
            x0, y0 = xy((x1, y2))
            d.line([x0, y0 + cell, x0 + cell, y0 + cell], fill=SHELF, width=4)
        else:
            x0, y0 = xy((x2, y1))
            d.line([x0, y0, x0, y0 + cell], fill=SHELF, width=4)

    f = frames[min(t, len(frames) - 1)]
    n = len(f['pos'])
    for k in range(n):  # goals
        gx, gy = xy(grid.cell(int(grid_goals[k])))
        d.rectangle([gx + 3, gy + 3, gx + cell - 3, gy + cell - 3], outline=ROBOT[k], width=3)
    for k in range(n):  # trails
        pts = [xy(grid.cell(int(fr['pos'][k]))) for fr in frames[max(0, t - 12):t + 1] if not fr['done'][k]
               or fr is frames[0]]
        for i in range(1, len(pts)):
            a = 0.25 + 0.75 * i / len(pts)
            col = tuple(int(c * a + 255 * (1 - a)) for c in ROBOT[k])
            (x0, y0), (x1, y1) = pts[i - 1], pts[i]
            d.line([x0 + cell / 2, y0 + cell / 2, x1 + cell / 2, y1 + cell / 2], fill=col, width=max(2, cell // 8))
    collided = {i for c in f['conflicts'] for i in c[:2]}
    for k in range(n):  # robots
        if f['done'][k]:
            continue
        x0, y0 = xy(grid.cell(int(f['pos'][k])))
        cx, cy, r = x0 + cell / 2, y0 + cell / 2, cell * 0.36
        if f['coord'][k]:
            d.ellipse([cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5], outline=RING, width=3)
        if k in collided:
            d.ellipse([cx - r - 9, cy - r - 9, cx + r + 9, cy + r + 9], outline=BAD, width=3)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ROBOT[k], outline=SURFACE, width=2)
        d.text((cx, cy), str(k + 1), fill='white', font=f_num, anchor='mm')

    d.text((4, 6), title, fill=INK, font=f_title)
    d.text((4, 26), f'{n_states:,.0f} states per robot', fill=INK2, font=f_small)
    status = []
    for k in range(n):
        done_at = next((fr['t'] for fr in frames[:t + 1] if fr['done'][k]), None)
        status.append(f'R{k + 1}: ' + (f'arrived t={done_at}' if done_at else ('coordinating' if f['coord'][k] else 'moving')))
    d.text((4, top + h + 6), f't = {min(t, len(frames) - 1)}   ' + '   '.join(status), fill=INK2, font=f_small)
    return img


grid_goals = None


def main():
    global grid_goals
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--map', default='warehouse')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--methods', nargs='+', default=['indep', 'js', 'cq'])
    ap.add_argument('--p-fail', type=float, default=0.0, help='failed-move probability during the replay')
    ap.add_argument('--replay-seed', type=int, default=0, help='rng for failed moves')
    ap.add_argument('--max-steps', type=int, default=60)
    ap.add_argument('--cell', type=int, default=30)
    ap.add_argument('--ms', type=int, default=350, help='milliseconds per frame')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    grid = Grid.load(args.map)
    panels = []
    for m in args.methods:
        path = Path(args.run) / args.map / f'{m}_seed{args.seed}.npz'
        policies, meta = load_run(path)
        env = MultiRobotEnv(grid, p_fail=args.p_fail, seed=args.replay_seed)
        grid_goals = env.goals
        frames = rollout(env, policies, max_steps=args.max_steps)
        n_states = np.mean([len(grid.free) + len(p.qj) for p in policies]) if m == 'cq' else \
            (len(grid.free) ** len(policies) if m == 'js' else len(grid.free))
        steps = [next((fr['t'] for fr in frames if fr['done'][k]), None) for k in range(env.n)]
        print(f'{m}: steps to goal {steps}, collisions {sum(len(fr["conflicts"]) for fr in frames)}')
        panels.append((m, frames, n_states))
    n_frames = max(len(fr) for _, fr, _ in panels) + 4   # hold the last frame
    images = []
    for t in range(n_frames):
        tiles = [draw_panel(grid, fr, t, LABELS[m], args.cell, ns) for m, fr, ns in panels]
        gap = 16
        W = sum(im.width for im in tiles) + gap * (len(tiles) - 1)
        canvas = Image.new('RGB', (W, tiles[0].height), SURFACE)
        x = 0
        for im in tiles:
            canvas.paste(im, (x, 0))
            x += im.width + gap
        images.append(canvas)
    out = Path(args.out or Path(args.run) / 'figures' / f'replay_{args.map}_seed{args.seed}_{"_".join(args.methods)}.gif')
    out.parent.mkdir(parents=True, exist_ok=True)
    durations = [args.ms] * (len(images) - 1) + [2500]
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    # Still image: the first moment a CQ robot coordinates, if any.
    still = 8
    for m, fr, _ in panels:
        hit = next((f['t'] for f in fr if m == 'cq' and any(f['coord'])), None)
        if hit is not None:
            still = hit
            break
    images[min(len(images) - 1, still)].save(out.with_suffix('.png'))
    print(f'wrote {out} ({len(images)} frames) and a still {out.with_suffix(".png")}')


if __name__ == '__main__':
    main()
