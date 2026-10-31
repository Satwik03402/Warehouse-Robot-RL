"""Train a NEW five-action PPO policy for the complete pickup-delivery mission."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from warehouse.mission import MissionEnv

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--timesteps',type=int,default=500000)
    args=parser.parse_args()
    check_env(MissionEnv(final_goal_assist=False),warn=True)
    out=Path('outputs');out.mkdir(exist_ok=True)
    env=Monitor(MissionEnv(final_goal_assist=False),filename=str(out/'mission_monitor.csv'))
    model=PPO('MlpPolicy',env,verbose=1,seed=42,learning_rate=3e-4,n_steps=1024,
              batch_size=256,gamma=.995,ent_coef=.01,tensorboard_log=str(out/'mission_tb'),device='cpu')
    model.learn(args.timesteps)
    model.save(str(out/'ppo_mission_unassisted'))
    env.close()
    print('Saved outputs/ppo_mission_unassisted.zip')
if __name__=='__main__':main()
