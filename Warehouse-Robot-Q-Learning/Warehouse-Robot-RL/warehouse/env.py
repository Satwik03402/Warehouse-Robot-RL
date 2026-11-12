"""Lightweight, deterministic grid navigation environment for RL."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .layout import LAYOUT

# 0:right, 1:up, 2:left, 3:down
DELTAS = ((1, 0), (0, 1), (-1, 0), (0, -1))

class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, max_steps=250, render_mode=None):
        super().__init__()
        self.layout = LAYOUT
        self.action_space = spaces.Discrete(4)
        # robot x/y, goal dx/dy, obstacle bits in four adjacent cells
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.robot = self.layout.start
        self.steps = 0

    def _obs(self):
        x, y = self.robot
        gx, gy = self.layout.goal
        occupied = [float(self.layout.blocked((x + dx, y + dy))) for dx, dy in DELTAS]
        return np.asarray([x / (self.layout.width - 1), y / (self.layout.height - 1),
                           (gx - x + self.layout.width - 1) / (2 * (self.layout.width - 1)),
                           (gy - y + self.layout.height - 1) / (2 * (self.layout.height - 1)),
                           *occupied], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.robot = self.layout.start
        self.steps = 0
        return self._obs(), {"position": self.robot}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        old = self.robot
        dx, dy = DELTAS[int(action)]
        candidate = (old[0] + dx, old[1] + dy)
        collision = self.layout.blocked(candidate)
        if not collision:
            self.robot = candidate
        self.steps += 1
        gx, gy = self.layout.goal
        old_dist = abs(gx - old[0]) + abs(gy - old[1])
        new_dist = abs(gx - self.robot[0]) + abs(gy - self.robot[1])
        success = self.robot == self.layout.goal
        reward = -0.1 + 0.5 * (old_dist - new_dist) - (2.0 if collision else 0.0)
        if success:
            reward += 100.0
        terminated = bool(success)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        return self._obs(), float(reward), terminated, truncated, {
            "position": self.robot, "collision": collision, "success": success,
            "steps": self.steps,
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        scale = 32
        canvas = np.full((self.layout.height * scale, self.layout.width * scale, 3), 238, dtype=np.uint8)
        def fill(cell, rgb):
            x, y = cell
            canvas[(self.layout.height - 1 - y) * scale:(self.layout.height - y) * scale,
                   x * scale:(x + 1) * scale] = rgb
        for c in self.layout.shelves:
            fill(c, (103, 119, 138))
        fill(self.layout.goal, (56, 176, 99))
        fill(self.layout.start, (238, 187, 72))
        fill(self.robot, (55, 110, 223))
        return canvas
