"""Train one reusable PPO navigation policy across randomly sampled start/goal cells."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from warehouse.goal_env import GoalNavigationEnv

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--timesteps',type=int,default=300000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    check_env(GoalNavigationEnv(),warn=True)
    out=Path('outputs');out.mkdir(exist_ok=True)
    env=Monitor(GoalNavigationEnv(),filename=str(out/'goal_monitor.csv'))
    env.reset(seed=args.seed)
    model=PPO('MlpPolicy',env,verbose=1,seed=args.seed,n_steps=1024,batch_size=256,
              learning_rate=3e-4,gamma=.99,ent_coef=.015,device='cpu',
              tensorboard_log=str(out/'goal_tb'))
    model.learn(total_timesteps=args.timesteps)
    model.save(out/'ppo_goal_navigation')
    env.close()
    print('Saved outputs/ppo_goal_navigation.zip')
if __name__=='__main__':main()
