"""Run ONE goal-conditioned policy for every mission leg, with automatic service waits."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from warehouse.goal_env import GoalNavigationEnv
from warehouse.mission import MissionEnv, WAIT
from warehouse.mission_view import play2d,play3d

def run(model,seed=42,max_steps=220):
    mission=MissionEnv(max_steps=max_steps,final_goal_assist=False)
    nav=GoalNavigationEnv(random_goals=False)
    mission.reset(seed=seed)
    frames=[dict(mission._info(False),action=None)]
    total=0; collisions=0;stage_routes=[]
    for _ in range(max_steps):
        if mission.stage<2 and mission.robot==mission.target:
            action=WAIT
        else:
            nav.reset(options={'start':mission.robot,'goal':mission.target})
            action=int(model.predict(nav._obs(),deterministic=True)[0])
        stage_before=mission.stage
        _,r,done,trunc,info=mission.step(action)
        collisions+=int(info['collision']);total+=r
        frames.append(dict(info,action=action))
        if info['stage']!=frames[-2]['stage']:stage_routes.append((stage_before,info['steps']))
        if done or trunc:break
    nav.close();mission.close()
    return frames,total,collisions,stage_routes

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('check','2d','3d'),default='check')
    p.add_argument('--model',default='outputs/ppo_goal_navigation.zip')
    p.add_argument('--max-steps',type=int,default=220)
    args=p.parse_args()
    if not Path(args.model).is_file():p.error('Train first with python -m scripts.train_goal_ppo')
    model=PPO.load(args.model,device='cpu')
    frames,total,collisions,stages=run(model,max_steps=args.max_steps)
    info=frames[-1]
    print(f'Goal-PPO mission success={info["success"]} stage={info["stage"]} steps={info["steps"]} reward={total:.1f} collisions={collisions}')
    print('Controller: GOAL-CONDITIONED PPO on ALL 3 legs; automatic pickup/drop-off waits; NO A* action assist')
    print('Final cell:',info['position'],'| stages:',stages)
    if args.mode=='2d':play2d(frames)
    elif args.mode=='3d':play3d(frames)
if __name__=='__main__':main()
