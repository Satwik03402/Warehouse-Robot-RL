"""Train fixed-start, goal-conditioned or full-mission tabular Q-learning."""
import argparse
from pathlib import Path
from warehouse.env import WarehouseEnv
from warehouse.goal_env import GoalNavigationEnv
from warehouse.mission import MissionEnv
from warehouse.qlearning import QTable, train

DEFAULTS = {'fixed': 15000, 'goal': 100000, 'mission': 40000}
FILES = {'fixed': 'q_fixed.npz', 'goal': 'q_goal.npz', 'mission': 'q_mission.npz'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=tuple(DEFAULTS), default='goal')
    ap.add_argument('--episodes', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--alpha', type=float, default=0.25)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--epsilon-end', type=float, default=0.05)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    episodes = args.episodes or DEFAULTS[args.task]
    if episodes <= 0: ap.error('--episodes must be positive')
    if not 0 < args.alpha <= 1: ap.error('--alpha must be in (0, 1]')
    if not 0 <= args.gamma <= 1: ap.error('--gamma must be in [0, 1]')
    envs = {'fixed': lambda: WarehouseEnv(max_steps=100),
            'goal': lambda: GoalNavigationEnv(max_steps=100),
            'mission': lambda: MissionEnv(max_steps=220, final_goal_assist=False)}
    env = envs[args.task]()
    agent = QTable(args.task, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    score = train(agent, env, episodes, seed=args.seed,
                  epsilon_end=args.epsilon_end, report_every=max(1, episodes//20))
    path = args.output or Path('outputs')/FILES[args.task]
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(path)
    env.close()
    print(f'Saved {path} | final training-window success={score:.1%}')

if __name__ == '__main__': main()
