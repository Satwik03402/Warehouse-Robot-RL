"""Tabular off-policy Q-learning for discrete warehouse coordinates (no PPO or A* controller)."""
import numpy as np
from .layout import LAYOUT

class QTable:
    def __init__(self, kind, alpha=0.25, gamma=0.99, seed=42):
        shapes = {
            'fixed': (LAYOUT.width, LAYOUT.height, 4),
            'goal': (LAYOUT.width, LAYOUT.height, LAYOUT.width, LAYOUT.height, 4),
            'mission': (3, LAYOUT.width, LAYOUT.height, 6, 5),
        }
        if kind not in shapes: raise ValueError(f'Unknown kind: {kind}')
        self.kind = kind
        self.q = np.zeros(shapes[kind], dtype=np.float32)
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.default_rng(seed)

    def state(self, env):
        x, y = env.robot
        if self.kind == 'fixed': return (x, y)
        if self.kind == 'goal': return (x, y, *env.goal)
        return (min(env.stage, 2), x, y, min(env.wait_count, 5))

    def act(self, state, epsilon=0.0, env=None):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q.shape[-1]))
        values = self.q[state]
        # Deterministic tie breaking for reproducible evaluation.
        return int(np.argmax(values))

    def update(self, state, action, reward, next_state, terminal):
        # Treat truncation as terminal in the finite training rollout.
        target = reward if terminal else reward + self.gamma * float(np.max(self.q[next_state]))
        self.q[state + (int(action),)] += self.alpha * (target - self.q[state + (int(action),)])

    def save(self, path):
        np.savez_compressed(path, q=self.q, kind=self.kind, alpha=self.alpha, gamma=self.gamma)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            obj = cls(str(z['kind']), alpha=float(z['alpha']), gamma=float(z['gamma']))
            if obj.q.shape != z['q'].shape: raise ValueError('Q-table shape mismatch')
            obj.q[:] = z['q']
        return obj


def train(agent, env, episodes, seed=42, epsilon_start=1.0, epsilon_end=0.05,
          decay_fraction=0.8, report_every=1000):
    success_window = []
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        epsilon = epsilon_end + (epsilon_start-epsilon_end)*max(0., 1. - episode/max(1,episodes*decay_fraction))
        state = agent.state(env)
        while True:
            action = agent.act(state, epsilon)
            _, reward, done, truncated, info = env.step(action)
            next_state = agent.state(env)
            agent.update(state, action, reward, next_state, done or truncated)
            state = next_state
            if done or truncated: break
        success_window.append(bool(info['success']))
        if report_every and (episode+1) % report_every == 0:
            print(f'Episode {episode+1}/{episodes} | recent success {np.mean(success_window[-report_every:]):.1%} | epsilon {epsilon:.3f}', flush=True)
    return float(np.mean(success_window[-min(1000,episodes):]))
