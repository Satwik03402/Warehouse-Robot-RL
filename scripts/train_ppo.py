"""Train PPO with a fixed layout, start and goal. Run from project root."""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from warehouse.env import WarehouseEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=100)
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error('--timesteps must be positive')
    out = Path('outputs')
    out.mkdir(exist_ok=True)
    check_env(WarehouseEnv(max_steps=args.max_steps), warn=True)
    env = Monitor(WarehouseEnv(max_steps=args.max_steps), filename=str(out / 'train_monitor.csv'))
    model = PPO('MlpPolicy', env, verbose=1, seed=args.seed,
                tensorboard_log=str(out / 'tensorboard'),
                learning_rate=3e-4, n_steps=1024, batch_size=256,
                gamma=0.99, ent_coef=0.01, device='cpu')
    checkpoint = CheckpointCallback(save_freq=50_000, save_path=str(out / 'checkpoints'), name_prefix='ppo_warehouse')
    model.learn(total_timesteps=args.timesteps, callback=checkpoint)
    model.save(str(out / 'ppo_warehouse'))
    env.close()
    print(f'Saved final policy to {out / "ppo_warehouse.zip"}')
    print('Next: python -m scripts.evaluate_ppo')


if __name__ == '__main__':
    main()
