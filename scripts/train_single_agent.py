"""Train the single-agent Q-learning prior (CQ-learning's Q_k) on the paper's maps.

One table per (map, robot, seed). Each robot learns alone from its own start to its own goal.
Defaults follow the paper: alpha 0.05, fixed epsilon 0.1, 10,000 episodes, 10 runs.
The paper assumes an *optimal* prior without saying how it was trained. Optimistic initial
values (--q-init 20, the goal reward) make every run find the shortest path within that budget;
with --q-init 0, 5 of 20 CMU runs settle on a detour.

    python3 -m scripts.train_single_agent                      # all maps, 10 seeds
    python3 -m scripts.train_single_agent --maps ttg tr --seeds 3
"""
import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from marl.grid import Grid, available_maps
from marl.qlearning import greedy_rollout, optimal_q, policy_agreement, train
from marl.single_env import RewardConfig, SingleRobotEnv


def run_one(job):
    name, agent, seed, args = job
    grid = Grid.load(name)
    start, goal = grid.agents[agent]
    rewards = RewardConfig(noise_sigma=args['noise_sigma'])
    env = SingleRobotEnv(grid, start, goal, rewards, max_steps=args['max_steps'])
    t0 = time.perf_counter()
    res = train(env, episodes=args['episodes'], alpha=args['alpha'], epsilon=args['epsilon'],
                gamma=args['gamma'], seed=seed, start_mode=args['start_mode'], q_init=args['q_init'])
    seconds = time.perf_counter() - t0
    q_star = optimal_q(env, gamma=args['gamma'])
    path, reached = greedy_rollout(res.q, env)
    shortest = int(grid.distances_to(goal)[grid.idx(start)])
    on_path = path[:-1] if reached else path
    metrics = dict(
        map=name, agent=agent + 1, seed=seed, start=list(start), goal=list(goal),
        n_states=grid.n_states, shortest_path=shortest, greedy_steps=len(path) - 1,
        reached_goal=bool(reached), optimal_path=bool(reached and len(path) - 1 == shortest),
        policy_agreement=policy_agreement(res.q, q_star, env.goal),
        max_q_error_on_path=float(np.abs(res.q[on_path] - q_star[on_path]).max()),
        last100_episode_length=float(res.episode_lengths[-100:].mean()),
        train_seconds=round(seconds, 2),
    )
    out = Path(args['out']) / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f'agent{agent + 1}_seed{seed}.npz',
        q=res.q, q_star=q_star, reward_count=res.reward_count, reward_mean=res.reward_mean,
        reward_m2=res.reward_m2, episode_lengths=res.episode_lengths,
        episode_success=res.episode_success,
        free_cells=np.array(grid.free), start=np.array(start), goal=np.array(goal),
        meta=json.dumps(dict(map=name, agent=agent + 1, **res.config, metrics=metrics)),
    )
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--maps', nargs='+', default=available_maps(), choices=available_maps())
    ap.add_argument('--seeds', type=int, default=10, help='independent runs per robot (paper: 10)')
    ap.add_argument('--episodes', type=int, default=10_000)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--epsilon', type=float, default=0.1)
    ap.add_argument('--gamma', type=float, default=0.95, help='not stated in the paper')
    ap.add_argument('--max-steps', type=int, default=500, help='episode time limit (not stated in the paper)')
    ap.add_argument('--start-mode', choices=('fixed', 'random'), default='fixed')
    ap.add_argument('--q-init', type=float, default=20.0, help='initial Q-value (optimistic by default)')
    ap.add_argument('--noise-sigma', type=float, default=0.0, help='reward noise std (paper: unspecified)')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())))
    ap.add_argument('--out', default='outputs/single_agent')
    args = ap.parse_args()
    params = {k: getattr(args, k) for k in ('episodes', 'alpha', 'epsilon', 'gamma', 'max_steps',
                                            'start_mode', 'q_init', 'noise_sigma', 'out')}
    jobs = [(m, a, s, params) for m in args.maps for a in range(len(Grid.load(m).agents))
            for s in range(args.seeds)]
    print(f'{len(jobs)} runs on {args.workers} workers: {params}', flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, jobs))
    out = Path(args.out)
    with open(out / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / 'config.json').write_text(json.dumps(dict(params, seeds=args.seeds, maps=args.maps), indent=2))

    print(f'\nDone in {time.perf_counter() - t0:.0f}s. Per robot, over {args.seeds} seeds:')
    print(f'{"map":5} {"robot":>5} {"cells":>5} {"shortest":>8} {"greedy steps":>13} {"optimal path":>12} '
          f'{"policy agree":>12} {"train s":>7}')
    for m in args.maps:
        for a in sorted({r['agent'] for r in rows if r['map'] == m}):
            rs = [r for r in rows if r['map'] == m and r['agent'] == a]
            steps = np.array([r['greedy_steps'] for r in rs])
            print(f'{m:5} {a:5d} {rs[0]["n_states"]:5d} {rs[0]["shortest_path"]:8d} '
                  f'{steps.mean():7.1f}±{steps.std():<5.1f} {sum(r["optimal_path"] for r in rs):>7d}/{len(rs):<4d} '
                  f'{np.mean([r["policy_agreement"] for r in rs]):12.2f} '
                  f'{np.mean([r["train_seconds"] for r in rs]):7.1f}')
    print(f'\nSaved tables and reward statistics under {out}/<map>/, summary in {out}/runs.csv')


if __name__ == '__main__':
    main()
