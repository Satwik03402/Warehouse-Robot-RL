"""Replay a learned PPO route in the existing 2D or 3D visualizer."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from warehouse.env import WarehouseEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('2d', '3d'), default='2d')
    parser.add_argument('--model', default='outputs/ppo_warehouse.zip')
    parser.add_argument('--max-steps', type=int, default=100)
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error('Model not found; run python -m scripts.train_ppo first')
    env = WarehouseEnv(max_steps=args.max_steps)
    model = PPO.load(args.model, device='cpu')
    obs, _ = env.reset(seed=42)
    path = [env.robot]
    collisions = 0
    for _ in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        collisions += int(info['collision'])
        path.append(info['position'])
        if done or truncated:
            break
    env.close()
    print(f'PPO replay: success={info["success"]}; steps={info["steps"]}; collisions={collisions}')
    if not info['success']:
        print('Policy did not reach goal. Replay still shows what it learned; try more training.')
    if args.mode == '2d':
        from warehouse.view2d import play
    else:
        from warehouse.view3d import play
    play(path)


if __name__ == '__main__':
    main()
