import numpy as np

from marl.arena import MultiRobotEnv
from marl.baselines import evaluate, train
from marl.cq import CQLearner, welch_p
from marl.grid import Grid
from marl.qlearning import train as train_alone
from marl.single_env import RewardConfig, SingleRobotEnv

NOISY = RewardConfig(noise_sigma=1.0)


def priors(grid, starts, goals, episodes=4000):
    out = []
    for k, (s, g) in enumerate(zip(starts, goals)):
        env = SingleRobotEnv(grid, s, g, NOISY)
        out.append(train_alone(env, episodes=episodes, seed=k, start_mode='random', q_init=20.0))
    return out


def test_welch_one_sided_only_flags_worse_rewards():
    assert welch_p(-10, 1, 20, 0, 1, 5000) < 1e-6          # much worse -> significant
    assert welch_p(10, 1, 20, 0, 1, 5000) > 0.99           # better is not "dangerous"
    assert 0.3 < welch_p(0, 1, 20, 0, 1, 5000) < 0.7       # same mean -> about 0.5
    assert welch_p(-10, 1, 20, 0, 1, 5000, one_sided=False) < 1e-6


def test_no_joint_states_when_robots_never_meet():
    g = Grid(['......', '######', '......'])
    starts, goals = [(0, 2), (0, 0)], [(5, 2), (5, 0)]
    env = MultiRobotEnv(g, starts=starts, goals=goals, rewards=NOISY, max_steps=50)
    learners = [CQLearner(i, g.n_states, 2, p) for i, p in enumerate(priors(g, starts, goals))]
    train(env, learners, episodes=1000, seed=0)
    assert [len(l.qj) for l in learners] == [0, 0] and [l.n_states for l in learners] == [12, 12]
    ev = evaluate(env, learners)
    assert ev['success'] and ev['steps_per_robot'] == [5, 5]


def test_head_on_meeting_is_detected_near_the_other_robot_only():
    g = Grid(['.......', '.......'])      # robots swap ends of the bottom row, so they meet
    starts, goals = [(0, 0), (6, 0)], [(6, 0), (0, 0)]
    env = MultiRobotEnv(g, starts=starts, goals=goals, rewards=NOISY, max_steps=60)
    learners = [CQLearner(i, g.n_states, 2, p) for i, p in enumerate(priors(g, starts, goals))]
    curves = train(env, learners, episodes=3000, seed=0)
    assert sum(len(l.qj) for l in learners) > 0
    assert sum(len(l.qj) for l in learners) < 0.2 * 2 * g.n_states ** 2   # small part of the joint space
    # A collision needs the other robot within two cells; the 1% test allows rare false alarms.
    dist = [abs(np.subtract(g.cell(js[0]), g.cell(js[1]))).sum()
            for l in learners for js in l.qj if js[1] < g.n_states]
    assert np.mean(np.array(dist) <= 2) >= 0.8
    assert curves['collisions'][-500:].mean() < curves['collisions'][:500].mean()


def test_local_table_is_frozen_and_qj_bootstraps_from_it():
    g = Grid(['.....'])
    p = priors(g, [(0, 0)], [(4, 0)], episodes=500)[0]
    l = CQLearner(0, g.n_states, 2, p, alpha=0.5)
    before = l.q_local.copy()
    js = (1, 2)
    l.qj[js] = l.q_local[1].copy()
    l.update(np.array([1, 2]), np.array([False, False]), 0, -10.0, np.array([1, 2]), np.array([False, False]))
    expected = 0.5 * before[1, 0] + 0.5 * (-10.0 + 0.95 * before[1].max())
    assert np.isclose(l.qj[js][0], expected)
    assert np.array_equal(l.q_local, before)
