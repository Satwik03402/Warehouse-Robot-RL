"""Goal-conditioned navigation: randomized reachable start/target, no A* action assistance."""
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .layout import LAYOUT
from .env import DELTAS

class GoalNavigationEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, max_steps=100, random_goals=True):
        super().__init__()
        self.layout = LAYOUT
        self.max_steps = max_steps
        self.random_goals = random_goals
        self.action_space = spaces.Discrete(4)
        # normalized current xy, target xy, relative dx/dy, 4 blocked bits, 4x4 local obstacles
        self.observation_space = spaces.Box(0, 1, shape=(26,), dtype=np.float32)
        self.free = [(x,y) for x in range(self.layout.width) for y in range(self.layout.height)
                     if not self.layout.blocked((x,y))]
        self.reset(seed=0)

    def _dist(self, start, target):
        if start == target: return 0
        visited = {start}; q = deque([(start,0)])
        while q:
            pos, d = q.popleft()
            for dx,dy in DELTAS:
                p = (pos[0]+dx,pos[1]+dy)
                if self.layout.blocked(p) or p in visited: continue
                if p == target:return d+1
                visited.add(p);q.append((p,d+1))
        return None

    def _obs(self):
        x,y=self.robot; gx,gy=self.goal
        w,h=self.layout.width-1,self.layout.height-1
        adjacent=[float(self.layout.blocked((x+dx,y+dy))) for dx,dy in DELTAS]
        # 4x4 local occupancy patch offsets (-1,0,1,2) on each axis
        patch=[float(self.layout.blocked((x+dx,y+dy))) for dy in (-1,0,1,2) for dx in (-1,0,1,2)]
        return np.array([x/w,y/h,gx/w,gy/h,(gx-x+w)/(2*w),(gy-y+h)/(2*h),*adjacent,*patch],dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options=options or {}
        if 'start' in options and 'goal' in options:
            start,goal=tuple(options['start']),tuple(options['goal'])
        elif self.random_goals:
            while True:
                indices=self.np_random.choice(len(self.free),size=2,replace=False)
                start,goal=self.free[int(indices[0])],self.free[int(indices[1])]
                d=self._dist(start,goal)
                if d is not None and d>=3 and d<=self.max_steps-5:break
        else:
            start,goal=self.layout.start,self.layout.goal
        if self.layout.blocked(start) or self.layout.blocked(goal) or self._dist(start,goal) is None:
            raise ValueError(f'Invalid/unreachable start-goal: {start} {goal}')
        self.robot,self.goal=start,goal
        self.steps=0
        return self._obs(),{'position':self.robot,'goal':self.goal}

    def step(self, action):
        if not self.action_space.contains(action):raise ValueError(f'Invalid action {action}')
        old=self.robot
        dx,dy=DELTAS[int(action)]
        nxt=(old[0]+dx,old[1]+dy)
        collision=self.layout.blocked(nxt)
        if not collision:self.robot=nxt
        self.steps+=1
        # potential shaping using true maze distances (for learning only, never action selection)
        before=self._dist(old,self.goal)
        after=self._dist(self.robot,self.goal)
        reward=-0.1+1.0*(before-after)-(2.0 if collision else 0)
        done=self.robot==self.goal
        if done:reward+=50.0
        trunc=self.steps>=self.max_steps and not done
        return self._obs(),float(reward),done,trunc,{'position':self.robot,'goal':self.goal,'success':done,'collision':collision,'steps':self.steps}
