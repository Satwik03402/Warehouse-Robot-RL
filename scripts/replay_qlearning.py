"""Replay learned Q-table using the existing 2D/3D PyBullet viewers."""
import argparse
from pathlib import Path
from warehouse.env import WarehouseEnv
from warehouse.goal_env import GoalNavigationEnv
from warehouse.mission import MissionEnv, WAIT
from warehouse.qlearning import QTable
from warehouse.mission_view import play2d, play3d


def run_mission(agent, max_steps=220):
    mission = MissionEnv(max_steps=max_steps, final_goal_assist=False)
    nav = GoalNavigationEnv(random_goals=False) if agent.kind == 'goal' else None
    mission.reset(seed=42)
    frames = [dict(mission._info(False), action=None)]
    total = collisions = 0
    stages = []
    for _ in range(max_steps):
        if agent.kind == 'goal':
            if mission.stage < 2 and mission.robot == mission.target:
                action = WAIT
            else:
                nav.reset(options={'start': mission.robot, 'goal': mission.target})
                action = agent.act(agent.state(nav))
        else:
            action = agent.act(agent.state(mission))
        previous = mission.stage
        _, reward, done, truncated, info = mission.step(action)
        total += reward
        collisions += int(info['collision'])
        frames.append(dict(info, action=action))
        if mission.stage != previous: stages.append((previous, mission.steps))
        if done or truncated: break
    mission.close()
    if nav is not None: nav.close()
    return frames, total, collisions, stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=('fixed', 'goal', 'mission'), default='goal')
    ap.add_argument('--model', type=Path)
    ap.add_argument('--mode', choices=('check', '2d', '3d'), default='check')
    ap.add_argument('--max-steps', type=int, default=220)
    args = ap.parse_args()
    path = args.model or Path('outputs') / f'q_{args.task}.npz'
    if not path.is_file(): ap.error(f'Missing {path}; train first with python -m scripts.train_qlearning --task {args.task}')
    agent = QTable.load(path)
    if agent.kind != args.task: ap.error(f'Model task {agent.kind} does not match {args.task}')
    if args.task == 'fixed':
        env = WarehouseEnv(max_steps=args.max_steps)
        env.reset(seed=42)
        route = [env.robot]
        collisions = total = 0
        for _ in range(args.max_steps):
            action = agent.act(agent.state(env))
            _, reward, done, truncated, info = env.step(action)
            route.append(env.robot)
            collisions += int(info['collision']); total += reward
            if done or truncated: break
        print(f'Fixed Q-learning: success={info["success"]} steps={info["steps"]} collisions={collisions} reward={total:.1f}')
        env.close()
        if args.mode != 'check':
            if args.mode == '2d': from warehouse.view2d import play
            else: from warehouse.view3d import play
            play(route)
    else:
        frames, reward, collisions, stages = run_mission(agent, args.max_steps)
        info = frames[-1]
        print(f'{args.task.capitalize()} Q-learning mission: success={info["success"]} '
              f'stage={info["stage"]} steps={info["steps"]} reward={reward:.1f} collisions={collisions}')
        print(f'Stage transitions: {stages}; final cell: {info["position"]}; no A* action assistance')
        if args.mode == '2d': play2d(frames)
        elif args.mode == '3d': play3d(frames)

if __name__ == '__main__': main()
