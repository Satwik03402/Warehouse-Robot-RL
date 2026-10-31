"""Pickup -> wait five simulated seconds -> drop-off -> wait -> final goal."""
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .layout import LAYOUT
from .env import DELTAS

PICKUP = (2, 9)
DROPOFF = (13, 2)
STATIONS = (PICKUP, DROPOFF, LAYOUT.goal)
STAGE_NAMES = ('PICKUP', 'DROPOFF', 'FINAL GOAL', 'COMPLETE')
WAIT = 4
WAIT_SECONDS = 5  # one WAIT action represents one simulated second

class MissionEnv(gym.Env):
    """One deterministic mission. Five consecutive WAIT actions required at each station."""
    metadata = {'render_modes': []}
    def __init__(self, max_steps=220, final_goal_assist=True):
        super().__init__()
        self.layout = LAYOUT
        self.action_space = spaces.Discrete(5)
        # pos XY, target relative XY, blocked neighbor bits, stage one-hot (3), progress
        self.observation_space = spaces.Box(0, 1, (12,), dtype=np.float32)
        self.max_steps = max_steps
        self.final_goal_assist = final_goal_assist
        self.reset()

    @property
    def target(self):
        return STATIONS[min(self.stage, 2)]

    def _obs(self):
        x, y = self.robot
        tx, ty = self.target
        w, h = self.layout.width - 1, self.layout.height - 1
        obstacles = [float(self.layout.blocked((x+dx, y+dy))) for dx,dy in DELTAS]
        stage = [float(self.stage == i) for i in range(3)]
        return np.asarray([x/w, y/h, (tx-x+w)/(2*w), (ty-y+h)/(2*h),
                           *obstacles, *stage, self.wait_count/WAIT_SECONDS], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.robot = self.layout.start
        self.stage = 0
        self.wait_count = 0
        self.steps = 0
        self.carrying = False
        return self._obs(), self._info(False)

    def _info(self, collision):
        return {'position':self.robot, 'stage':STAGE_NAMES[self.stage],
                'wait_seconds':self.wait_count, 'carrying':self.carrying,
                'collision':collision, 'success':self.stage==3, 'steps':self.steps}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Action out of range: {action}')
        # Station service is automatic once the robot reaches pickup/drop-off.
        # The agent cannot depart until five WAIT ticks have completed.
        if self.stage < 2 and self.robot == self.target:
            action = WAIT
        # HYBRID DEMO ONLY: A* handles the whole final leg, not just the last cell.
        # Disable for scientifically valid, unassisted PPO evaluation.
        if self.final_goal_assist and self.stage == 2 and self.robot != self.layout.goal:
            route = shortest_path(self.robot, self.layout.goal, self.layout)
            next_cell = route[1]
            delta = (next_cell[0] - self.robot[0], next_cell[1] - self.robot[1])
            action = DELTAS.index(delta)
        old = self.robot
        old_target = self.target
        collision = False
        reward = -0.1
        if int(action) == WAIT:
            if self.stage < 2 and self.robot == self.target:
                self.wait_count += 1
                reward += 1.0  # encourage correct waiting
                if self.wait_count == WAIT_SECONDS:
                    reward += 30.0
                    if self.stage == 0:
                        self.carrying = True
                    else:
                        self.carrying = False
                    self.stage += 1
                    self.wait_count = 0
            else:
                reward -= 1.0
                self.wait_count = 0
        else:
            dx,dy = DELTAS[int(action)]
            nxt = (old[0]+dx, old[1]+dy)
            collision = self.layout.blocked(nxt)
            if collision:
                reward -= 2.0
            else:
                self.robot = nxt
            if self.stage < 2:
                self.wait_count = 0  # moving off station interrupts service
            # Reward actual path progress around shelves, not straight-line distance.
            old_dist = len(shortest_path(old, old_target, self.layout)) - 1
            new_dist = len(shortest_path(self.robot, old_target, self.layout)) - 1
            reward += 0.5 * (old_dist - new_dist)
        if self.stage == 2 and self.robot == self.layout.goal:
            self.stage = 3
            reward += 100.0
        self.steps += 1
        terminated = self.stage == 3
        truncated = self.steps >= self.max_steps and not terminated
        return self._obs(), float(reward), terminated, truncated, self._info(collision)


def shortest_path(start, goal, layout=LAYOUT):
    queue = deque([start]); parent = {start:None}
    while queue:
        pos=queue.popleft()
        if pos == goal:
            path=[]
            while pos is not None:
                path.append(pos); pos=parent[pos]
            return path[::-1]
        for dx,dy in DELTAS:
            nxt=(pos[0]+dx, pos[1]+dy)
            if not layout.blocked(nxt) and nxt not in parent:
                parent[nxt]=pos; queue.append(nxt)
    raise ValueError(f'Unreachable station {start} -> {goal}')


def baseline_actions():
    reverse = {d:i for i,d in enumerate(DELTAS)}
    actions=[]; pos=LAYOUT.start
    for i, station in enumerate(STATIONS):
        route=shortest_path(pos,station)
        actions.extend(reverse[(b[0]-a[0],b[1]-a[1])] for a,b in zip(route,route[1:]))
        if i < 2:
            actions.extend([WAIT]*WAIT_SECONDS)
        pos=station
    return actions


def rollout(actions, max_steps=220):
    env=MissionEnv(max_steps=max_steps)
    obs,info=env.reset()
    frames=[dict(info,action=None)]
    total=0
    for action in actions:
        obs,r,done,trunc,info=env.step(int(action))
        total+=r
        frames.append(dict(info,action=int(action)))
        if done or trunc: break
    env.close()
    return frames,total
