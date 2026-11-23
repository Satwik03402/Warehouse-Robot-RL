"""Load trained policies from scripts.train_baselines output and play them back.

    policies, meta = load_run('outputs/table1/tr/cq_seed0.npz')
    frames = rollout(env, policies)            # one dict per time step
"""
import json

import numpy as np

from .qlearning import greedy


class SavedPolicy:
    """Greedy policy of one robot, rebuilt from a saved run (Indep, JS or CQ)."""

    def __init__(self, i, method, data, n_cells, n_robots):
        self.i, self.method, self.n_cells = i, method, n_cells
        self.others = [j for j in range(n_robots) if j != i]
        self.q = data[f'q{i + 1}']
        self.qj = {}
        if method == 'cq':
            keys, vals = data[f'qj{i + 1}_keys'], data[f'qj{i + 1}_values']
            self.qj = {tuple(int(x) for x in k): v for k, v in zip(keys, vals)}

    def key(self, pos, done):
        return (int(pos[self.i]),) + tuple(self.n_cells if done[j] else int(pos[j]) for j in self.others)

    def row(self, pos, done):
        k = self.key(pos, done)
        if self.method == 'indep':
            return self.q[k[0]]
        if self.method == 'js':
            return self.q[k]
        r = self.qj.get(k)
        return self.q[k[0]] if r is None else r

    def in_coordination(self, pos, done):
        return self.method == 'cq' and self.key(pos, done) in self.qj

    def act(self, pos, done, rng=None, epsilon=0.0):
        return greedy(self.row(pos, done))


def load_run(path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data['meta']))
    n = sum(1 for k in data.files if k.startswith('q') and k[1:].isdigit())
    n_cells = len(data['free_cells'])
    return [SavedPolicy(i, meta['method'], data, n_cells, n) for i in range(n)], meta


def rollout(env, policies, max_steps=None):
    """Greedy episode; returns frames with positions, actions, rewards, conflicts."""
    pos, done = env.reset(), env.done.copy()
    frames = [dict(t=0, pos=pos.copy(), done=done.copy(), coord=[p.in_coordination(pos, done) for p in policies],
                   actions=None, reward=None, conflicts=[], failed=None)]
    for t in range(1, (max_steps or env.max_steps) + 1):
        acts = [policies[i].act(pos, done) if not done[i] else 0 for i in range(env.n)]
        pos, r, done, truncated, info = env.step(acts)
        frames.append(dict(t=t, pos=pos.copy(), done=done.copy(),
                           coord=[p.in_coordination(pos, done) for p in policies],
                           actions=acts, reward=r.copy(), conflicts=info['conflicts'], failed=info['failed'].copy()))
        if done.all() or truncated:
            break
    return frames
