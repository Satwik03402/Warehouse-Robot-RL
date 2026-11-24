"""CQ-learning (De Hauwere, Vrancx & Nowe, AAMAS 2010, Algorithm 1).

Each robot starts from a prior learned alone (`marl.qlearning.train`): an optimal local
table Q_k(s_k, a) and a model of the immediate reward for every (s_k, a) (Welford mean,
variance and count). During the multi-robot phase the local table is frozen. The robot
looks for places where the other robots change its rewards:

1. Every reward observed for (s_k, a_k) goes into a sliding window. Once the window holds
   `min_samples` rewards, a Welch t-test compares it with the prior's reward model.
   If they differ at `confidence`, (s_k, a_k) is flagged.
2. Rewards are also kept per joint state js = (own cell, other robots' cells). For a flagged
   (s_k, a_k), every js seen there with at least `min_samples` rewards is tested against
   the prior the same way. A js that differs is marked *dangerous* and gets a row in the
   joint table Q^j_k(js, .), copied from Q_k(s_k, .). A js that does not differ is marked
   *safe* for that action and is not tested again (re-testing a busy state thousands of
   times would turn the 1% false-alarm rate into near-certain false alarms).

In a dangerous js the robot acts on Q^j_k, otherwise on Q_k. Only Q^j_k is updated:
    Q^j(js, a) <- (1 - alpha) Q^j(js, a) + alpha [r + gamma max_a' Q_k(s'_k, a')]
The target bootstraps from the local table, as in the paper (r alone after reaching the goal).

Choices the paper leaves open, agreed with the user: reward noise sigma 1 (set in the
environment), window of the last 20 rewards, at least 10 samples per test, 99% confidence,
Q^j initialised from Q_k. Found necessary to keep false alarms near the nominal 1%:
- the test is one-sided (rewards *worse* than alone), since another robot can only hurt;
- a window is tested once per 10 new rewards, not after every reward;
- joint samples are collected only after (s_k, a_k) is flagged, as in Algorithm 1, so the
  second test never reuses the rewards that triggered the first;
- each (s_k, a_k, js) is tested once and then stays dangerous or safe.
"""
from collections import deque

import numpy as np
from scipy.special import stdtr

from .baselines import JointStateLearner
from .qlearning import greedy


def welch_p(m1, v1, n1, m2, v2, n2, one_sided=True):
    """p-value of Welch's t-test from summary stats.

    one_sided: H1 is "mean 1 is lower than mean 2" (rewards got worse), else two-sided.
    """
    a, b = v1 / n1, v2 / n2
    se2 = a + b
    if se2 <= 0:
        if m1 == m2:
            return 1.0
        return 0.0 if (m1 < m2 or not one_sided) else 1.0
    t = (m1 - m2) / np.sqrt(se2)
    df = se2 ** 2 / (a ** 2 / (n1 - 1) + b ** 2 / (n2 - 1))
    return float(stdtr(df, t)) if one_sided else float(2 * stdtr(df, -abs(t)))


class _Window(deque):
    """Last `maxlen` rewards plus how many arrived since the last test."""

    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)
        self.fresh = 0

    def append(self, x):
        super().append(x)
        self.fresh += 1


def _stats(window):
    n = len(window)
    m = sum(window) / n
    v = sum((x - m) ** 2 for x in window) / (n - 1)
    return m, v, n


class CQLearner(JointStateLearner):
    def __init__(self, i, n_cells, n_robots, prior, gamma=0.95, alpha=0.05,
                 window=20, min_samples=10, confidence=0.99, test_every=10, one_sided=True,
                 joint_bootstrap=False):
        """`prior` is a marl.qlearning.TrainResult for this robot learned alone."""
        self.i, self.gamma, self.alpha = i, gamma, alpha
        self.others = [j for j in range(n_robots) if j != i]
        self.n_cells = n_cells
        self.q_local = prior.q.copy()
        self.prior_config = prior.config
        self.prior_mean = prior.reward_mean
        self.prior_var = prior.reward_var
        self.prior_n = prior.reward_count
        self.window, self.min_samples, self.alpha_test = window, min_samples, 1 - confidence
        self.test_every, self.one_sided = test_every, one_sided
        self.joint_bootstrap = joint_bootstrap
        self.recent = {}                      # (s, a) -> last `window` rewards
        self.joint_recent = {}                # (s, a, js) -> last `window` rewards
        self.flagged = np.zeros_like(self.q_local, dtype=bool)
        self.qj = {}                          # dangerous js -> Q^j row (4,)
        self.safe = set()                     # (s, a, js) tested once and found safe
        self.danger_log = []                  # (js, s, a, episode-free step counter) when marked
        self.updates = 0

    @property
    def n_states(self):
        """State-space size as in Table 1: local states plus joint states in Q^j."""
        return self.n_cells + len(self.qj)

    def row(self, pos, done):
        js = self.key(pos, done)
        r = self.qj.get(js)
        return self.q_local[js[0]] if r is None else r

    def act(self, pos, done, rng=None, epsilon=0.0):
        if rng is not None and rng.random() < epsilon:
            return int(rng.integers(4))
        return greedy(self.row(pos, done), rng)

    def _test(self, samples, s, a):
        """None if there is not enough new data yet, else whether rewards are worse than the prior."""
        n0 = self.prior_n[s, a]
        if len(samples) < self.min_samples or n0 < 2 or samples.fresh < self.test_every:
            return None
        samples.fresh = 0
        m, v, n = _stats(samples)
        p = welch_p(m, v, n, self.prior_mean[s, a], self.prior_var[s, a], n0, self.one_sided)
        return p < self.alpha_test

    def update(self, pos, done, a, r, next_pos, next_done):
        self.updates += 1
        js = self.key(pos, done)
        s = js[0]
        row = self.qj.get(js)
        if row is not None:
            if next_done[self.i]:
                target = r
            elif self.joint_bootstrap:   # variant: Q^j(js') if js' is dangerous, else Q_k(s')
                target = r + self.gamma * self.row(next_pos, next_done).max()
            else:                        # paper: always the local table
                target = r + self.gamma * self.q_local[int(next_pos[self.i])].max()
            row[a] += self.alpha * (target - row[a])

        w = self.recent.get((s, a))
        if w is None:
            w = self.recent[(s, a)] = _Window(self.window)
        w.append(r)
        if not self.flagged[s, a]:
            if self._test(w, s, a):
                self.flagged[s, a] = True    # joint samples are collected from now on
            return
        key = (s, a, js)
        jw = self.joint_recent.get(key)
        if jw is None:
            jw = self.joint_recent[key] = _Window(self.window)
        jw.append(r)
        if row is None and key not in self.safe:
            verdict = self._test(jw, s, a)
            if verdict:
                self.qj[js] = self.q_local[s].copy()
                self.danger_log.append((js, s, a, self.updates))
            elif verdict is not None:
                self.safe.add(key)

    def coordination_cells(self):
        """Count of dangerous joint states per own cell (for heatmaps)."""
        counts = np.zeros(self.n_cells, dtype=np.int64)
        for js in self.qj:
            counts[js[0]] += 1
        return counts
