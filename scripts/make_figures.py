"""Report figures from scripts.train_baselines output.

    python3 -m scripts.make_figures outputs/warehouse_p0                 # one run directory
    python3 -m scripts.make_figures outputs/warehouse_p0 outputs/warehouse_p0.1 outputs/warehouse_p0.2

Writes into <run>/figures/:
  coordination_<map>.png   where CQ placed its joint states (summed over seeds), per robot
  curves_<map>.png         training collisions and steps per episode (moving average, mean of seeds)
  size_steps_<map>.png     state-space size vs greedy steps per method
and, with several runs, figures/pfail_<map>.png in the first run comparing failure rates.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from marl.grid import Grid  # noqa: E402

COLORS = {'indep': '#2a78d6', 'js': '#eb6834', 'cq': '#1baf7a'}   # fixed categorical order
LABELS = {'indep': 'Indep', 'js': 'JS', 'cq': 'CQ'}
ROBOT_COLORS = ['#2a78d6', '#eb6834']
INK, INK2, MUTED, GRID_INK, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
HEAT = LinearSegmentedColormap.from_list('heat', ['#eaf2fc', '#2a78d6', '#0d2f5c'])  # one hue, light -> dark

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': GRID_INK, 'axes.labelcolor': INK2, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'legend.frameon': False,
})


def load_rows(run):
    return list(csv.DictReader(open(Path(run) / 'runs.csv')))


def runs_of(run, name, method):
    return sorted((Path(run) / name).glob(f'{method}_seed*.npz'))


def draw_grid(ax, grid, values=None, vmax=None, cmap=HEAT):
    """Free cells coloured by `values` (None = blank), blocked cells dark gray, y up."""
    h, w = grid.height, grid.width
    img = np.full((h, w, 4), 0.0)
    for y in range(h):
        for x in range(w):
            img[h - 1 - y, x] = matplotlib.colors.to_rgba('#3d3d3a')
    norm = matplotlib.colors.Normalize(0, vmax or 1)
    for s, (x, y) in enumerate(grid.free):
        v = None if values is None else values[s]
        img[h - 1 - y, x] = matplotlib.colors.to_rgba(SURFACE) if not v else cmap(norm(v))
    ax.imshow(img, extent=(-0.5, w - 0.5, -0.5, h - 0.5), interpolation='nearest')
    for x in range(w + 1):
        ax.axvline(x - 0.5, color=GRID_INK, lw=0.5)
    for y in range(h + 1):
        ax.axhline(y - 0.5, color=GRID_INK, lw=0.5)
    for (a, b) in grid.walls:  # thin walls between cells
        (x1, y1), (x2, y2) = sorted([tuple(a), tuple(b)])
        if x1 == x2:
            ax.plot([x1 - 0.5, x1 + 0.5], [y1 + 0.5] * 2, color='#3d3d3a', lw=3)
        else:
            ax.plot([x1 + 0.5] * 2, [y1 - 0.5, y1 + 0.5], color='#3d3d3a', lw=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def mark_robot(ax, k, start, goal):
    """Start = filled numbered dot; goal = square outline in the robot's colour (drawn larger,
    so a goal that is the other robot's start stays visible around its dot)."""
    c = ROBOT_COLORS[k]
    ax.scatter(*goal, s=330, marker='s', facecolor='none', edgecolor=c, linewidth=2.2, zorder=5)
    ax.scatter(*start, s=120, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=6)
    ax.text(*start, str(k + 1), color='white', ha='center', va='center', fontsize=7, weight='bold', zorder=7)


def fig_coordination(run, name, figdir):
    files = runs_of(run, name, 'cq')
    if not files:
        return
    grid = Grid.load(name)
    n = len(grid.agents)
    counts = [sum(np.load(f)[f'coord_cells{k + 1}'] for f in files) for k in range(n)]
    vmax = max(c.max() for c in counts)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n * grid.width / max(grid.width, grid.height) + 1, 4.4))
    for k, ax in enumerate(np.atleast_1d(axes)):
        draw_grid(ax, grid, counts[k], vmax)
        for j, (s, g) in enumerate(grid.agents):
            mark_robot(ax, j, s, g)
        ax.set_title(f'Robot {k + 1}: {counts[k].sum() / len(files):.0f} joint states per run', loc='left')
    sm = plt.cm.ScalarMappable(cmap=HEAT, norm=matplotlib.colors.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(f'dangerous joint states at this cell\n(summed over {len(files)} seeds)', color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f'Where CQ-learning coordinates: {name}', x=0.02, ha='left', color=INK, fontsize=12)
    fig.text(0.02, 0.02, 'Dot = robot start, square outline = its goal (same colour). Dark = blocked (shelves/walls).',
             color=MUTED, fontsize=8.5)
    fig.savefig(figdir / f'coordination_{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def moving_average(x, w=200):
    c = np.cumsum(np.insert(x.astype(float), 0, 0))
    return (c[w:] - c[:-w]) / w


def fig_curves(run, name, figdir, methods):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for meth in methods:
        files = runs_of(run, name, meth)
        if not files:
            continue
        coll = np.mean([moving_average(np.load(f)['train_collisions']) for f in files], axis=0)
        steps = np.mean([moving_average(np.load(f)['train_steps']) for f in files], axis=0)
        x = np.arange(len(coll)) + 200
        for ax, y in ((a1, coll), (a2, steps)):
            ax.plot(x, y, color=COLORS[meth], lw=2, label=LABELS[meth])
    a1.set_title('Collisions per training episode', loc='left')
    a2.set_title('Steps per robot per training episode', loc='left')
    a2.set_yscale('log')
    for ax in (a1, a2):
        ax.set_xlabel('episode (moving average of 200)')
        ax.grid(axis='y', color=GRID_INK, lw=0.6)
    a1.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(figdir / f'curves_{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_size_steps(rows, name, figdir, methods):
    rs = {m: [r for r in rows if r['map'] == name and r['method'] == m] for m in methods}
    rs = {m: v for m, v in rs.items() if v}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ms = list(rs)
    size = [np.mean([float(r['n_states']) for r in rs[m]]) for m in ms]
    ok = [[float(r['mean_steps']) for r in rs[m] if r['success'] == 'True'] for m in ms]
    succ = [np.mean([float(r['success_rate']) for r in rs[m]]) for m in ms]
    y = np.arange(len(ms))
    a1.barh(y, size, color=[COLORS[m] for m in ms], height=0.55)
    a1.set_xscale('log')
    a1.set_xlim(10, max(size) * 4)
    for i, v in enumerate(size):
        a1.text(v * 1.15, i, f'{v:,.0f}', va='center', color=INK, fontsize=9)
    a1.set_title('States per robot (log scale)', loc='left')
    a2.barh(y, [np.mean(o) if o else 0 for o in ok], color=[COLORS[m] for m in ms], height=0.55)
    for i, (o, s) in enumerate(zip(ok, succ)):
        a2.text((np.mean(o) if o else 0) + 0.5, i, f'{np.mean(o):.1f}  ({s:.0%} success)' if o else 'no success',
                va='center', color=INK, fontsize=9)
    a2.set_title('Greedy steps per robot (successful runs)', loc='left')
    a2.set_xlim(0, max(np.mean(o) for o in ok if o) * 1.6)
    for ax in (a1, a2):
        ax.set_yticks(y, [LABELS[m] for m in ms])
        ax.invert_yaxis()
        ax.grid(axis='x', color=GRID_INK, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figdir / f'size_steps_{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_pfail(runs, name, figdir, methods):
    ps, data = [], {m: ([], []) for m in methods}
    for run in runs:
        cfg = json.loads((Path(run) / 'config.json').read_text())
        ps.append(cfg['p_fail'])
        rows = [r for r in load_rows(run) if r['map'] == name]
        for m in methods:
            rr = [r for r in rows if r['method'] == m]
            data[m][0].append(np.mean([float(r['success_rate']) for r in rr]) if rr else np.nan)
            data[m][1].append(np.mean([float(r['mean_steps']) for r in rr if float(r['success_rate']) > 0])
                              if rr else np.nan)
    order = np.argsort(ps)
    ps = np.array(ps)[order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for m in methods:
        for ax, vals in ((a1, data[m][0]), (a2, data[m][1])):
            v = np.array(vals)[order]
            ax.plot(ps, v, color=COLORS[m], lw=2, marker='o', ms=7, mec=SURFACE, mew=2, label=LABELS[m])
            if ax is a2:   # success rates often coincide at 1.0, so only the steps panel gets end labels
                ax.text(ps[-1] + 0.006, v[-1], LABELS[m], color=INK2, va='center', fontsize=9)
    a1.set_title('Greedy success rate', loc='left')
    a1.set_ylim(0, 1.05)
    a2.set_title('Greedy steps per robot', loc='left')
    for ax in (a1, a2):
        ax.set_xlabel('probability a move fails')
        ax.set_xticks(ps)
        ax.set_xlim(ps[0] - 0.01, ps[-1] + 0.04)
        ax.grid(axis='y', color=GRID_INK, lw=0.6)
    a1.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(figdir / f'pfail_{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    runs = sys.argv[1:] or ['outputs/warehouse_p0']
    for run in runs:
        rows = load_rows(run)
        figdir = Path(run) / 'figures'
        figdir.mkdir(exist_ok=True)
        maps = list(dict.fromkeys(r['map'] for r in rows))
        methods = [m for m in COLORS if any(r['method'] == m for r in rows)]
        for name in maps:
            fig_coordination(run, name, figdir)
            fig_curves(run, name, figdir, methods)
            fig_size_steps(rows, name, figdir, methods)
        print(f'figures in {figdir}')
    if len(runs) > 1:
        rows = load_rows(runs[0])
        methods = [m for m in COLORS if any(r['method'] == m for r in rows)]
        for name in dict.fromkeys(r['map'] for r in rows):
            fig_pfail(runs, name, Path(runs[0]) / 'figures', methods)


if __name__ == '__main__':
    main()
