"""Compare saved PPO policy with a shortest-path A* and random baseline."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from warehouse.env import WarehouseEnv
from warehouse.layout import LAYOUT
from warehouse.planner import astar, path_actions


def evaluate(policy, episodes, max_steps):
    env = WarehouseEnv(max_steps=max_steps)
    stats = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=1000 + episode)
        total, collisions = 0., 0
        for step in range(max_steps):
            action = policy(obs, env)
            obs, reward, done, truncated, info = env.step(int(action))
            total += reward
            collisions += int(info['collision'])
            if done or truncated:
                break
        stats.append((int(info['success']), info['steps'], collisions, total))
    env.close()
    s = np.array(stats, dtype=float)
    return dict(success_rate=float(s[:, 0].mean()), mean_steps=float(s[:, 1].mean()),
                mean_collisions=float(s[:, 2].mean()), mean_reward=float(s[:, 3].mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--model', default='outputs/ppo_warehouse.zip')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    model_path = Path(args.model)
    if not model_path.is_file():
        parser.error(f'Missing model: {model_path}. Run train_ppo first.')
    model = PPO.load(str(model_path), device='cpu')
    rng = np.random.default_rng(123)
    astar_actions = path_actions(astar(LAYOUT))
    print(f'A* shortest path: {len(astar_actions)} steps')

    # No independent episode variation yet: this is a fixed-layout sanity check,
    # not a generalization or statistically independent performance evaluation.
    for name, policy in [
        ('PPO', lambda obs, env: model.predict(obs, deterministic=True)[0]),
        ('Random', lambda obs, env: rng.integers(0, env.action_space.n)),
        ('A*', lambda obs, env: astar_actions[min(env.steps, len(astar_actions)-1)]),
    ]:
        stats = evaluate(policy, args.episodes, args.max_steps)
        print(f'{name:6} | success={stats["success_rate"]:.0%} | '
              f'steps={stats["mean_steps"]:.1f} | collisions={stats["mean_collisions"]:.1f} | '
              f'reward={stats["mean_reward"]:.1f}')
    print('Note: all episodes share the same start, goal, and shelves.')


if __name__ == '__main__':
    main()
