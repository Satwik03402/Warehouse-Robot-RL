"""Single-agent tabular Q-learning: CQ-learning's prior Q_k and expected-reward model.

Paper settings (Section 5): learning rate 0.05, fixed epsilon-greedy with epsilon = 0.1,
10,000 training iterations (taken here to mean episodes). The paper does not state gamma.

Besides Q, training records Welford statistics (count, mean, M2) of the immediate rewards
seen for each (state, action). CQ-learning's first t-test compares rewards observed with
other robots present against this single-agent model.
"""
from dataclasses import dataclass, field

import numpy as np

from .single_env import SingleRobotEnv


@dataclass
class TrainResult:
    q: np.ndarray
    reward_count: np.ndarray
    reward_mean: np.ndarray
    reward_m2: np.ndarray
    episode_lengths: np.ndarray
    episode_success: np.ndarray
    config: dict = field(default_factory=dict)

    @property
    def reward_var(self):
        """Unbiased sample variance per (state, action); NaN where fewer than two samples."""
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(self.reward_count > 1, self.reward_m2 / (self.reward_count - 1), np.nan)


def greedy(q_row, rng=None):
    """Argmax with random tie-breaking if `rng` is given, else the lowest index."""
    if rng is None:
        return int(np.argmax(q_row))
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[0] if len(best) == 1 else rng.choice(best))


def train(env: SingleRobotEnv, episodes=10_000, alpha=0.05, epsilon=0.1, gamma=0.95,
          seed=0, start_mode='fixed', q_init=0.0):
    """Q-learning with fixed epsilon-greedy exploration.

    start_mode='fixed' starts every episode on the map's start cell (as in the paper);
    'random' starts on a uniformly random non-goal free cell (exploring starts).
    q_init > 0 gives optimistic initial values, which drive systematic exploration.
    """
    if start_mode not in ('fixed', 'random'):
        raise ValueError(f'unknown start_mode {start_mode!r}')
    rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(rng.integers(2**63))  # reward noise stream
    n = env.grid.n_states
    q = np.full((n, 4), float(q_init))
    count = np.zeros((n, 4), dtype=np.int64)
    mean = np.zeros((n, 4))
    m2 = np.zeros((n, 4))
    lengths = np.zeros(episodes, dtype=np.int64)
    success = np.zeros(episodes, dtype=bool)
    starts = [s for s in range(n) if s != env.goal]
    for ep in range(episodes):
        s = env.reset(None if start_mode == 'fixed' else rng.choice(starts))
        while True:
            a = int(rng.integers(4)) if rng.random() < epsilon else greedy(q[s], rng)
            s2, r, done, truncated, _ = env.step(a)
            # Welford update of the immediate-reward model.
            count[s, a] += 1
            d = r - mean[s, a]
            mean[s, a] += d / count[s, a]
            m2[s, a] += d * (r - mean[s, a])
            # Reaching the goal is terminal; a time-limit cut is not, so it still bootstraps.
            target = r if done else r + gamma * q[s2].max()
            q[s, a] += alpha * (target - q[s, a])
            s = s2
            if done or truncated:
                lengths[ep], success[ep] = env.steps, done
                break
    config = dict(episodes=episodes, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=seed,
                  start_mode=start_mode, q_init=q_init, max_steps=env.max_steps,
                  rewards=env.rewards.as_dict(), p_fail=env.p_fail)
    return TrainResult(q, count, mean, m2, lengths, success, config)


def optimal_q(env: SingleRobotEnv, gamma=0.95, tol=1e-10):
    """Exact Q* of the deterministic single-robot MDP by value iteration (noise-free rewards)."""
    g = env.grid
    r = np.array([[env.expected_reward(s, a) for a in range(4)] for s in range(g.n_states)])
    terminal = g.next_state == env.goal
    q = np.zeros((g.n_states, 4))
    while True:
        v = q.max(axis=1)
        new = r + np.where(terminal, 0.0, gamma * v[g.next_state])
        if np.abs(new - q).max() < tol:
            return new
        q = new


def greedy_rollout(q, env: SingleRobotEnv, start=None, max_steps=None):
    """Follow argmax Q from `start`; return (cells visited, reached goal)."""
    s = env.reset(start)
    path = [s]
    for _ in range(max_steps or env.max_steps):
        s, _, done, truncated, _ = env.step(greedy(q[s]))
        path.append(s)
        if done:
            return path, True
        if truncated:
            break
    return path, False


def policy_agreement(q, q_star, goal, tol=1e-9):
    """Fraction of non-goal states whose greedy action under q is optimal under q_star."""
    states = [s for s in range(len(q)) if s != goal]
    ok = [q_star[s, greedy(q[s])] >= q_star[s].max() - tol for s in states]
    return float(np.mean(ok))
