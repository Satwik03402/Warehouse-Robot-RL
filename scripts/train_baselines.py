"""Train Indep, JS and CQ-learning on the paper's maps (or the warehouse) and summarise like Table 1.

Settings follow the paper: alpha 0.05, fixed epsilon 0.1, 10,000 training episodes,
greedy evaluation, 10 runs. Indep and JS learn from scratch (Q starts at 0) with noise-free
rewards. CQ first learns each robot's prior alone (same p_fail, reward noise sigma
--noise-sigma, optimistic Q start, random starts, 30k episodes so Q_k is accurate in every
cell, not just on the path: stale optimistic values off the path make retreating look free
and cause livelocks), then learns with the other robot present using noisy
rewards, as in the paper. gamma and the episode time limit are not stated in the paper.

With --p-fail 0 the maps are deterministic and one greedy episode is the evaluation; with
p_fail > 0 greedy play is averaged over --eval-episodes episodes.

    python3 -m scripts.train_baselines                                  # Table 1: 5 maps x 3 methods x 10 seeds
    python3 -m scripts.train_baselines --maps ttg --methods cq --seeds 3
    python3 -m scripts.train_baselines --maps warehouse --p-fail 0.1 --out outputs/warehouse_p0.1
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
from marl.cq import CQLearner
from marl.grid import PAPER_MAPS, Grid, available_maps
from marl.qlearning import greedy_rollout, train as train_alone
from marl.single_env import RewardConfig, SingleRobotEnv

METHODS = ['indep', 'js', 'cq']

# Table 1 of De Hauwere et al. (2010): (#states, #steps mean, #steps std); collisions were 0 everywhere.
PAPER = {
    ('ttg', 'indep'): (25, 11.0, 0.0), ('ttg', 'js'): (625, 13.3, 1.1), ('ttg', 'cq'): (27, 10.6, 0.1),
    ('tr', 'indep'): (36, 9.0, 0.0), ('tr', 'js'): (1296, 12.1, 0.32), ('tr', 'cq'): (38.0, 9.05, 0.03),
    ('isr', 'indep'): (43, 7.7, 18.9), ('isr', 'js'): (1849, 5.8, 0.89), ('isr', 'cq'): (48.3, 5.0, 0.28),
    ('cit', 'indep'): (69, 18.9, 17.9), ('cit', 'js'): (4761, 15.4, 0.5), ('cit', 'cq'): (77.0, 11.2, 0.9),
    ('cmu', 'indep'): (133, 35.9, 2.8), ('cmu', 'js'): (17689, 47.7, 5.6), ('cmu', 'cq'): (140.5, 31.3, 0.9),
}


def make_learners(method, grid, env, a, seed):
    if method != 'cq':
        return [LEARNERS[method](i, grid.n_states, env.n, gamma=a['gamma'], alpha=a['alpha'],
                                 q_init=a['q_init']) for i in range(env.n)]
    learners = []
    noisy = RewardConfig(noise_sigma=a['noise_sigma'])
    for k, (start, goal) in enumerate(zip(env.starts, env.goals)):
        alone = SingleRobotEnv(grid, grid.cell(int(start)), grid.cell(int(goal)), noisy,
                               max_steps=a['max_steps'], p_fail=a['p_fail'])
        # The paper assumes each robot already acts optimally alone. Check that the learned prior's
        # greedy path (failures off) reaches the goal; if not, retrain with a fresh seed.
        shortest = int(grid.distances_to(grid.cell(int(goal)))[int(start)])
        for attempt in range(5):
            prior = train_alone(alone, episodes=a['prior_episodes'], alpha=a['alpha'], epsilon=a['epsilon'],
                                gamma=a['gamma'], seed=1000 * seed + k + 100_000 * attempt,
                                q_init=a['prior_q_init'], start_mode=a['prior_start_mode'])
            check = SingleRobotEnv(grid, grid.cell(int(start)), grid.cell(int(goal)), max_steps=4 * shortest)
            path, reached = greedy_rollout(prior.q, check)
            if reached:
                break
        prior.config.update(attempts=attempt + 1, greedy_steps=len(path) - 1, shortest=shortest)
        learners.append(CQLearner(k, grid.n_states, env.n, prior, gamma=a['gamma'], alpha=a['alpha'],
                                  window=a['window'], min_samples=a['min_samples'],
                                  confidence=a['confidence'], joint_bootstrap=a['joint_bootstrap']))
    return learners


def run_one(job):
    name, method, seed, a = job
    grid = Grid.load(name)
    rewards = RewardConfig(noise_sigma=a['noise_sigma'] if method == 'cq' else 0.0)
    env = MultiRobotEnv(grid, rewards=rewards, p_fail=a['p_fail'], max_steps=a['max_steps'])
    t0 = time.perf_counter()
    learners = make_learners(method, grid, env, a, seed)
    curves = train(env, learners, episodes=a['episodes'], epsilon=a['epsilon'], seed=seed)
    seconds = time.perf_counter() - t0
    # Report noise-free rewards; failures still come from the env's rng.
    env.rewards = RewardConfig()
    env.rng = np.random.default_rng(10_000 + seed)
    ev = evaluate(env, learners, episodes=1 if a['p_fail'] == 0 else a['eval_episodes'])
    out = Path(a['out']) / name
    out.mkdir(parents=True, exist_ok=True)
    extra = {}
    for i, l in enumerate(learners):
        if method == 'cq':
            keys = sorted(l.qj)
            extra[f'q{i + 1}'] = l.q_local
            extra[f'qj{i + 1}_keys'] = np.array(keys, dtype=np.int64).reshape(len(keys), env.n)
            extra[f'qj{i + 1}_values'] = np.array([l.qj[k] for k in keys]).reshape(len(keys), 4)
            extra[f'coord_cells{i + 1}'] = l.coordination_cells()
            extra[f'flagged{i + 1}'] = l.flagged
        else:
            extra[f'q{i + 1}'] = l.q
    np.savez_compressed(out / f'{method}_seed{seed}.npz', **extra,
                        **{f'train_{k}': v for k, v in curves.items()},
                        free_cells=np.array(grid.free),
                        meta=json.dumps(dict(map=name, method=method, seed=seed, **a, eval=ev)))
    row = dict(map=name, method=method, seed=seed,
               n_states=float(np.mean([l.n_states for l in learners])),
               success=ev['success'], success_rate=ev['success_rate'], mean_steps=ev['mean_steps'],
               collisions=ev['collisions'],
               **{f'steps_robot{i + 1}': s for i, s in enumerate(ev['steps_per_robot'])},
               train_last100_steps=float(curves['steps'][-100:].mean()),
               train_last100_collisions=float(curves['collisions'][-100:].mean()),
               train_seconds=round(seconds, 1))
    if method == 'cq':
        row['coord_states'] = [len(l.qj) for l in learners]
        row['prior_attempts'] = [l.prior_config['attempts'] for l in learners]
    return row


def summarise(rows, maps, methods, max_steps, paper=True):
    lines = [f'{"map":9} {"alg":5} {"#states":>8} {"success":>8} {"#coll":>6} {"#steps":>14} '
             f'{"steps, successful":>18}' + (' | paper #states  #steps' if paper else '')]
    for m in maps:
        for meth in methods:
            rs = [r for r in rows if r['map'] == m and r['method'] == meth]
            if not rs:
                continue
            st = np.array([r['mean_steps'] for r in rs])
            ok = np.array([r['mean_steps'] for r in rs if r['success']])
            ok_s = f'{ok.mean():.1f}±{ok.std():.1f}' if len(ok) else '-'
            ns = np.array([r['n_states'] for r in rs])
            line = (f'{m:9} {meth:5} {ns.mean():8.1f} {np.mean([r["success_rate"] for r in rs]):8.2f} '
                    f'{np.mean([r["collisions"] for r in rs]):6.2f} {st.mean():7.1f}±{st.std():<6.1f} {ok_s:>18}')
            p = PAPER.get((m, meth))
            if paper and p:
                line += f' | {p[0]:>13} {p[1]}±{p[2]}'
            lines.append(line)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--maps', nargs='+', default=PAPER_MAPS, choices=available_maps())
    ap.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--episodes', type=int, default=10_000)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--epsilon', type=float, default=0.1)
    ap.add_argument('--gamma', type=float, default=0.95)
    ap.add_argument('--q-init', type=float, default=0.0, help='Indep/JS initial Q')
    ap.add_argument('--prior-q-init', type=float, default=20.0, help='CQ prior initial Q (optimistic)')
    ap.add_argument('--prior-episodes', type=int, default=30_000, help='CQ prior: episodes alone')
    ap.add_argument('--prior-start-mode', choices=('fixed', 'random'), default='random',
                    help='CQ prior: random (exploring) starts make Q_k accurate off the path too')
    ap.add_argument('--max-steps', type=int, default=500)
    ap.add_argument('--p-fail', type=float, default=0.0, help='probability a move fails (robot stays)')
    ap.add_argument('--eval-episodes', type=int, default=100, help='greedy episodes when p_fail > 0')
    ap.add_argument('--noise-sigma', type=float, default=1.0, help='CQ reward noise std')
    ap.add_argument('--window', type=int, default=20, help='CQ: rewards kept per test')
    ap.add_argument('--min-samples', type=int, default=10, help='CQ: rewards needed before testing')
    ap.add_argument('--confidence', type=float, default=0.99, help='CQ: t-test confidence')
    ap.add_argument('--joint-bootstrap', action='store_true',
                    help='CQ variant: bootstrap Q^j from Q^j at the next joint state when it is dangerous')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())))
    ap.add_argument('--out', default='outputs/table1')
    args = ap.parse_args()
    params = {k: getattr(args, k) for k in ('episodes', 'alpha', 'epsilon', 'gamma', 'q_init', 'prior_q_init', 'prior_episodes', 'prior_start_mode',
                                            'max_steps', 'p_fail', 'eval_episodes', 'noise_sigma', 'window',
                                            'min_samples', 'confidence', 'joint_bootstrap', 'out')}
    cost = {'js': 2, 'cq': 1.5, 'indep': 1}
    jobs = sorted([(m, meth, s, params) for m in args.maps for meth in args.methods for s in range(args.seeds)],
                  key=lambda j: -Grid.load(j[0]).n_states * cost[j[1]])
    print(f'{len(jobs)} runs on {args.workers} workers: {params}', flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one, jobs):
            rows.append(row)
            print(f'  {row["map"]:9} {row["method"]:5} seed {row["seed"]}: success={row["success_rate"]:.2f} '
                  f'steps={row["mean_steps"]:.1f} coll={row["collisions"]:.2f} states={row["n_states"]:.0f} '
                  f'({row["train_seconds"]}s)', flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Keep earlier results for (map, method, seed) combinations not re-run now.
    done = {(r['map'], r['method'], r['seed']) for r in rows}
    if (out / 'runs.csv').exists():
        for r in csv.DictReader(open(out / 'runs.csv')):
            r['seed'] = int(r['seed'])
            for key in ('n_states', 'success_rate', 'mean_steps', 'collisions', 'steps_robot1', 'steps_robot2'):
                if r.get(key) not in (None, ''):
                    r[key] = float(r[key])
            r['success'] = r['success'] == 'True'
            if (r['map'], r['method'], r['seed']) not in done:
                rows.append(r)
    rows.sort(key=lambda r: (r['map'], r['method'], r['seed']))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out / 'config.json').write_text(json.dumps(dict(params, seeds=args.seeds, maps=args.maps,
                                                     methods=args.methods), indent=2))
    all_maps = list(dict.fromkeys(r['map'] for r in rows))
    all_methods = [m for m in METHODS if any(r['method'] == m for r in rows)]
    table = summarise(rows, all_maps, all_methods, args.max_steps, paper=args.p_fail == 0)
    (out / 'summary.txt').write_text(table + '\n')
    print(f'\nDone in {time.perf_counter() - t0:.0f}s. Greedy evaluation over {args.seeds} seeds '
          f'(#states = per robot; steps = mean per robot; a robot that never arrives counts {args.max_steps}):')
    print(table)
    print(f'\nSaved Q-tables and training curves under {out}/<map>/, summary in {out}/runs.csv')


if __name__ == '__main__':
    main()
