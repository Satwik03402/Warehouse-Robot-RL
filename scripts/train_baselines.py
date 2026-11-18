"""Train the paper's Indep and JS baselines on the paper's maps and summarise them like Table 1.

Settings follow the paper: alpha 0.05, fixed epsilon 0.1, 10,000 training episodes,
greedy evaluation, 10 runs, deterministic transitions, no reward noise. Q starts at 0
(learned from scratch); gamma and the episode time limit are not stated in the paper.

    python3 -m scripts.train_baselines                       # all maps, indep + js, 10 seeds
    python3 -m scripts.train_baselines --maps ttg --methods indep --seeds 3
"""
import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from marl.arena import MultiRobotEnv
from marl.baselines import LEARNERS, evaluate, train
from marl.grid import Grid, available_maps

# Table 1 of De Hauwere et al. (2010): (#states, #steps mean, #steps std), collisions were 0 everywhere.
PAPER = {
    ('ttg', 'indep'): (25, 11.0, 0.0), ('ttg', 'js'): (625, 13.3, 1.1),
    ('tr', 'indep'): (36, 9.0, 0.0), ('tr', 'js'): (1296, 12.1, 0.32),
    ('isr', 'indep'): (43, 7.7, 18.9), ('isr', 'js'): (1849, 5.8, 0.89),
    ('cit', 'indep'): (69, 18.9, 17.9), ('cit', 'js'): (4761, 15.4, 0.5),
    ('cmu', 'indep'): (133, 35.9, 2.8), ('cmu', 'js'): (17689, 47.7, 5.6),
}


def run_one(job):
    name, method, seed, a = job
    grid = Grid.load(name)
    env = MultiRobotEnv(grid, max_steps=a['max_steps'])
    learners = [LEARNERS[method](i, grid.n_states, env.n, gamma=a['gamma'], alpha=a['alpha'],
                                 q_init=a['q_init']) for i in range(env.n)]
    t0 = time.perf_counter()
    curves = train(env, learners, episodes=a['episodes'], epsilon=a['epsilon'], seed=seed)
    seconds = time.perf_counter() - t0
    ev = evaluate(env, learners)
    out = Path(a['out']) / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'{method}_seed{seed}.npz',
                        **{f'q{i + 1}': l.q for i, l in enumerate(learners)},
                        **{f'train_{k}': v for k, v in curves.items()},
                        meta=json.dumps(dict(map=name, method=method, seed=seed, **a, eval=ev)))
    return dict(map=name, method=method, seed=seed, n_states=learners[0].n_states,
                success=ev['success'], mean_steps=ev['mean_steps'], collisions=ev['collisions'],
                steps_robot1=ev['steps_per_robot'][0], steps_robot2=ev['steps_per_robot'][1],
                train_last100_steps=float(curves['steps'][-100:].mean()),
                train_last100_collisions=float(curves['collisions'][-100:].mean()),
                train_seconds=round(seconds, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--maps', nargs='+', default=available_maps(), choices=available_maps())
    ap.add_argument('--methods', nargs='+', default=list(LEARNERS), choices=list(LEARNERS))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--episodes', type=int, default=10_000)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--epsilon', type=float, default=0.1)
    ap.add_argument('--gamma', type=float, default=0.95)
    ap.add_argument('--q-init', type=float, default=0.0)
    ap.add_argument('--max-steps', type=int, default=500)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())))
    ap.add_argument('--out', default='outputs/baselines')
    args = ap.parse_args()
    params = {k: getattr(args, k) for k in ('episodes', 'alpha', 'epsilon', 'gamma', 'q_init', 'max_steps', 'out')}
    # Longest jobs first so the pool stays busy.
    jobs = sorted([(m, meth, s, params) for m in args.maps for meth in args.methods for s in range(args.seeds)],
                  key=lambda j: -Grid.load(j[0]).n_states * (2 if j[1] == 'js' else 1))
    print(f'{len(jobs)} runs on {args.workers} workers: {params}', flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one, jobs):
            rows.append(row)
            print(f'  {row["map"]:4} {row["method"]:5} seed {row["seed"]}: success={row["success"]} '
                  f'steps={row["mean_steps"]:.1f} coll={row["collisions"]} ({row["train_seconds"]}s)', flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r['map'], r['method'], r['seed']))
    with open(out / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / 'config.json').write_text(json.dumps(dict(params, seeds=args.seeds, maps=args.maps,
                                                     methods=args.methods), indent=2))
    print(f'\nDone in {time.perf_counter() - t0:.0f}s. Greedy evaluation over {args.seeds} seeds '
          f'(steps = mean per robot; a robot that never arrives counts {args.max_steps}):')
    print(f'{"map":4} {"alg":5} {"#states":>7} {"success":>8} {"#coll":>6} {"#steps":>14} {"steps, successful runs":>22} | paper #steps')
    for m in args.maps:
        for meth in args.methods:
            rs = [r for r in rows if r['map'] == m and r['method'] == meth]
            st = np.array([r['mean_steps'] for r in rs])
            ok = np.array([r['mean_steps'] for r in rs if r['success']])
            ok_s = f'{ok.mean():.1f}±{ok.std():.1f}' if len(ok) else '-'
            p = PAPER[(m, meth)]
            print(f'{m:4} {meth:5} {rs[0]["n_states"]:7d} {sum(r["success"] for r in rs):5d}/{len(rs):<2d} '
                  f'{np.mean([r["collisions"] for r in rs]):6.1f} {st.mean():7.1f}±{st.std():<6.1f} {ok_s:>22} | '
                  f'{p[1]}±{p[2]}')
    print(f'\nSaved Q-tables and training curves under {out}/<map>/, summary in {out}/runs.csv')


if __name__ == '__main__':
    main()
