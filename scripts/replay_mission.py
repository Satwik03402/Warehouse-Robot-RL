"""Replay the PPO mission policy; the A* demo is separately available."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from warehouse.mission import MissionEnv
from warehouse.mission_view import play2d,play3d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('check','2d','3d'),default='check')
    ap.add_argument('--model',default='outputs/ppo_mission.zip')
    ap.add_argument('--no-final-assist',action='store_true',help='Use PPO alone for the final leg (no A* assistance)')
    args=ap.parse_args()
    if not Path(args.model).is_file():ap.error('Train mission policy first: python -m scripts.train_mission')
    env=MissionEnv(final_goal_assist=not args.no_final_assist);model=PPO.load(args.model,device='cpu')
    obs,info=env.reset(seed=42)
    frames=[dict(info,action=None)];reward_sum=0
    for _ in range(env.max_steps):
        action,_=model.predict(obs,deterministic=True)
        obs,reward,done,trunc,info=env.step(int(action))
        reward_sum+=reward
        frames.append(dict(info,action=int(action)))
        if done or trunc:break
    print(f'Mission success={info["success"]} stage={info["stage"]} steps={info["steps"]} reward={reward_sum:.1f}')
    print('Controller: ' + ('PPO only' if args.no_final_assist else 'HYBRID (PPO pickup/dropoff + A* final goal)'))
    print('Final robot cell:', info['position'])
    if not info['success']:
        print('Mission timed out; remaining distance to final goal:',
              len(__import__('warehouse.mission', fromlist=['shortest_path']).shortest_path(info['position'], env.layout.goal)) - 1)
    if args.mode=='2d':play2d(frames)
    elif args.mode=='3d':play3d(frames)
if __name__=='__main__':main()
