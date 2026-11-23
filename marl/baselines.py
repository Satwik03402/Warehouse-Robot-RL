"""Paper baselines: independent Q-learners (Indep) and joint-state learners (JS).

Both learn from scratch with every robot present, using the paper's settings: learning
rate 0.05 and fixed epsilon-greedy with epsilon 0.1. Robots choose their own actions
independently. Indep sees only its own cell; JS sees every robot's cell (a robot that
has finished counts as "absent").

Learners share one interface, so the same loop will also run CQ-learning:
    act(pos, done, rng, epsilon) -> action for this robot
    update(pos, done, action, reward, next_pos, next_done)
    n_states -> state-space size as reported in the paper's Table 1
"""
import numpy as np

from .qlearning import greedy


class IndepLearner:
    def __init__(self, i, n_cells, n_robots, gamma=0.95, alpha=0.05, q_init=0.0):
        self.i, self.gamma, self.alpha = i, gamma, alpha
        self.q = np.full((n_cells, 4), float(q_init))
        self.n_states = n_cells

    def key(self, pos, done):
        return (int(pos[self.i]),)

    def act(self, pos, done, rng=None, epsilon=0.0):
        if rng is not None and rng.random() < epsilon:
            return int(rng.integers(4))
        return greedy(self.q[self.key(pos, done)], rng)

    def update(self, pos, done, a, r, next_pos, next_done):
        sa = self.key(pos, done) + (a,)
        if next_done[self.i]:
            target = r
        else:
            target = r + self.gamma * self.q[self.key(next_pos, next_done)].max()
        self.q[sa] += self.alpha * (target - self.q[sa])


class JointStateLearner(IndepLearner):
    """Q over (own cell, other robots' cells); an absent (finished) robot has index n_cells."""

    def __init__(self, i, n_cells, n_robots, gamma=0.95, alpha=0.05, q_init=0.0):
        super().__init__(i, n_cells, n_robots, gamma, alpha, q_init)
        self.others = [j for j in range(n_robots) if j != i]
        self.n_cells = n_cells
        self.q = np.full((n_cells,) + (n_cells + 1,) * len(self.others) + (4,), float(q_init))
        self.n_states = n_cells ** n_robots

    def key(self, pos, done):
        return (int(pos[self.i]),) + tuple(self.n_cells if done[j] else int(pos[j]) for j in self.others)


LEARNERS = {'indep': IndepLearner, 'js': JointStateLearner}


def run_episode(env, learners, rng=None, epsilon=0.0, learn=False):
    """Play one episode; returns per-robot steps-to-goal (or None), collisions, reward."""
    pos, done = env.reset(), env.done.copy()
    n = env.n
    steps_to_goal = [None] * n
    collisions = np.zeros(n, dtype=np.int64)
    total = np.zeros(n)
    while True:
        actions = [learners[i].act(pos, done, rng, epsilon) if not done[i] else 0 for i in range(n)]
        next_pos, r, next_done, truncated, info = env.step(actions)
        collisions += info['collided']
        total += r
        for i in info['active']:
            if learn:
                learners[i].update(pos, done, actions[i], r[i], next_pos, next_done)
            if info['reached'][i]:
                steps_to_goal[i] = env.steps
        pos, done = next_pos, next_done
        if done.all() or truncated:
            return steps_to_goal, collisions, total


def train(env, learners, episodes=10_000, epsilon=0.1, seed=0):
    """Train all robots together; returns per-episode (mean steps, collisions, success) curves."""
    rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(rng.integers(2**63))
    steps = np.zeros(episodes)
    collisions = np.zeros(episodes, dtype=np.int64)
    success = np.zeros(episodes, dtype=bool)
    for ep in range(episodes):
        to_goal, coll, _ = run_episode(env, learners, rng, epsilon, learn=True)
        success[ep] = all(t is not None for t in to_goal)
        steps[ep] = np.mean([t if t is not None else env.max_steps for t in to_goal])
        collisions[ep] = coll.sum()
    return dict(steps=steps, collisions=collisions, success=success)


def evaluate(env, learners, episodes=1):
    """Greedy play (epsilon = 0). With deterministic moves one episode suffices; with
    p_fail > 0 the result is averaged over `episodes` episodes (the env's rng drives failures).
    A robot that never arrives counts env.max_steps."""
    runs = [run_episode(env, learners) for _ in range(episodes)]
    per_robot = np.array([[t if t is not None else env.max_steps for t in tg] for tg, _, _ in runs])
    ok = np.array([all(t is not None for t in tg) for tg, _, _ in runs])
    coll = np.array([c.sum() for _, c, _ in runs])
    return dict(
        success=bool(ok.all()),
        success_rate=float(ok.mean()),
        steps_per_robot=per_robot.mean(axis=0).tolist(),
        mean_steps=float(per_robot.mean()),
        collisions=float(coll.mean()),
        reward=np.mean([tot for _, _, tot in runs], axis=0).tolist(),
    )
