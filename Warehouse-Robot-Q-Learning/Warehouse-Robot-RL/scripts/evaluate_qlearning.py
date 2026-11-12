"""Evaluate Q-learning with fixed-grid baselines or held-out random start/goal tests."""
import argparse
from pathlib import Path
import numpy as np
from warehouse.env import WarehouseEnv
from warehouse.goal_env import GoalNavigationEnv
from warehouse.mission import shortest_path
from warehouse.planner import astar, path_actions
from warehouse.layout import LAYOUT
from warehouse.qlearning import QTable
from scripts.replay_qlearning import run_mission

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=('fixed', 'goal', 'mission'), default='goal')
    ap.add_argument('--model', type=Path)
    ap.add_argument('--episodes', type=int, default=100)
    args = ap.parse_args()
    if args.episodes < 1: ap.error('--episodes must be positive')
    path = args.model or Path('outputs') / f'q_{args.task}.npz'
    if not path.exists(): ap.error(f'Missing model: {path}')
    model = QTable.load(path)
    if model.kind != args.task: ap.error('Wrong model task')
    if args.task == 'mission':
        frames, reward, collisions, stages = run_mission(model)
        print(f'Mission: success={frames[-1]["success"]}, steps={frames[-1]["steps"]}, '
              f'collisions={collisions}, reward={reward:.1f}, transitions={stages}')
        print('Only one fixed mission is configured; repeated episodes are not independent test cases.')
        return
    env = GoalNavigationEnv() if args.task == 'goal' else WarehouseEnv(max_steps=100)
    stats = []
    for i in range(args.episodes):
        env.reset(seed=100000 + i)
        optimum = len(shortest_path(env.robot, env.goal if args.task == 'goal' else LAYOUT.goal)) - 1
        collisions = 0
        for _ in range(env.max_steps):
            action = model.act(model.state(env))
            _, _, done, truncated, info = env.step(action)
            collisions += int(info['collision'])
            if done or truncated: break
        stats.append((int(info['success']), info['steps'], collisions,
                      info['steps'] - optimum if info['success'] else np.nan))
    s = np.asarray(stats, dtype=float)
    print(f'{args.task} Q-learning | episodes={args.episodes} | success={s[:,0].mean():.1%} '
          f'| steps={s[:,1].mean():.1f} | collisions={s[:,2].mean():.2f} '
          f'| extra steps on success={np.nanmean(s[:,3]) if np.any(s[:,0]) else float("nan"):.2f}')
    if args.task == 'fixed':
        print(f'Fixed-map A* shortest route: {len(path_actions(astar(LAYOUT)))} steps; same map/start/goal every run.')
    else:
        frames, reward, collisions, stages = run_mission(model)
        print(f'Full mission | success={frames[-1]["success"]} | steps={frames[-1]["steps"]} '
              f'| collisions={collisions} | reward={reward:.1f} | transitions={stages}')
    env.close()

if __name__ == '__main__': main()
