import argparse
import numpy as np
from stable_baselines3 import PPO
from warehouse.goal_env import GoalNavigationEnv
from warehouse.mission import shortest_path
from scripts.replay_goal_mission import run

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--model',default='outputs/ppo_goal_navigation.zip')
    args=ap.parse_args()
    model=PPO.load(args.model,device='cpu')
    env=GoalNavigationEnv(max_steps=100)
    successes=[];collisions=[];steps=[];excess=[]
    for i in range(args.episodes):
        obs,_=env.reset(seed=1000+i)
        optimum=len(shortest_path(env.robot,env.goal))-1
        nc=0
        for t in range(env.max_steps):
            action,_=model.predict(obs,deterministic=True)
            obs,_,done,trunc,info=env.step(int(action))
            nc+=int(info['collision'])
            if done or trunc:break
        successes.append(bool(info['success'])); collisions.append(nc);steps.append(info['steps'])
        if info['success']:excess.append(info['steps']-optimum)
    print(f'Random start/goal: {args.episodes} episodes | success={np.mean(successes)*100:.1f}% | steps={np.mean(steps):.1f} | collisions={np.mean(collisions):.2f} | extra steps on successes={np.mean(excess) if excess else float("nan"):.2f}')
    frames,rew,cc,stages=run(model)
    print(f'Full pickup/dropoff/goal mission: success={frames[-1]["success"]} | steps={frames[-1]["steps"]} | collisions={cc} | reward={rew:.1f} | stage transitions={stages}')
    env.close()
if __name__=='__main__':main()
