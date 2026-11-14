"""One robot alone on a paper map: the setting in which CQ-learning's prior Q_k is learned.

Rewards follow De Hauwere et al. (2010): +20 for reaching the goal, -1 for moving into a wall
(the robot stays put), 0 otherwise. The -10 robot-collision reward cannot occur with one robot,
but it lives in the same RewardConfig so the multi-robot arena uses identical numbers.
Transitions are deterministic, as in the paper.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    goal: float = 20.0
    wall: float = -1.0
    agent: float = -10.0
    step: float = 0.0
    # The paper draws CQ-learning rewards from a normal distribution centred on the values
    # above, "for implementation reasons" (the t-test needs variance), without giving sigma.
    # When sigma > 0, noise is added to every reward, including the 0 step reward.
    noise_sigma: float = 0.0

    def as_dict(self):
        return asdict(self)


class SingleRobotEnv:
    def __init__(self, grid, start, goal, rewards=RewardConfig(), max_steps=500, seed=None):
        self.grid = grid
        self.start = grid.idx(start)
        self.goal = grid.idx(goal)
        self.rewards = rewards
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = self.start
        self.steps = 0

    def reset(self, start=None):
        """Put the robot on `start` (a free-cell index), or on the map's start."""
        self.state = self.start if start is None else int(start)
        self.steps = 0
        return self.state

    def expected_reward(self, state, action):
        """Mean reward for (state, action), before any noise."""
        if self.grid.bumps[state, action]:
            return self.rewards.wall
        if self.grid.next_state[state, action] == self.goal:
            return self.rewards.goal
        return self.rewards.step

    def step(self, action):
        s = self.state
        bump = bool(self.grid.bumps[s, action])
        reward = self.expected_reward(s, action)
        if self.rewards.noise_sigma > 0:
            reward += self.rng.normal(0.0, self.rewards.noise_sigma)
        self.state = int(self.grid.next_state[s, action])
        self.steps += 1
        done = self.state == self.goal
        truncated = not done and self.steps >= self.max_steps
        return self.state, float(reward), done, truncated, {'bump': bump}
