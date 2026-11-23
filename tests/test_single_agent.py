import numpy as np
import pytest

from marl.grid import DELTAS, Grid, available_maps
from marl.qlearning import greedy_rollout, optimal_q, policy_agreement, train
from marl.single_env import RewardConfig, SingleRobotEnv

PAPER_FREE_CELLS = {'ttg': 25, 'tr': 36, 'isr': 43, 'cit': 69, 'cmu': 133}  # Table 1, Indep #states
RIGHT, UP, LEFT, DOWN = range(4)


def test_all_paper_maps_present_with_table1_cell_counts():
    assert set(PAPER_FREE_CELLS) <= set(available_maps())  # plus our own maps, e.g. warehouse
    for name, n in PAPER_FREE_CELLS.items():
        g = Grid.load(name)
        assert g.n_states == n
        assert len(g.agents) == 2


def test_index_round_trip():
    g = Grid.load('cmu')
    assert all(g.idx(g.cell(i)) == i for i in range(g.n_states))


def test_moves_are_symmetric():
    for name in PAPER_FREE_CELLS:
        g = Grid.load(name)
        for s in range(g.n_states):
            for a in range(4):
                if not g.bumps[s, a]:
                    assert g.next_state[g.next_state[s, a], (a + 2) % 4] == s


def test_ttg_tunnel_walls_block_vertical_moves():
    g = Grid.load('ttg')
    goal = g.idx((0, 2))
    assert g.bumps[goal, UP] and g.bumps[goal, DOWN] and g.bumps[goal, LEFT]
    assert not g.bumps[goal, RIGHT]
    # The tunnel opens only at x = 4, so both robots need 10 moves.
    dist = g.distances_to((0, 2))
    assert dist[g.idx((0, 4))] == 10 and dist[g.idx((0, 0))] == 10


def test_rewards_follow_paper():
    g = Grid.load('ttg')
    env = SingleRobotEnv(g, (0, 4), (0, 2))
    s = env.reset()
    s2, r, done, _, info = env.step(LEFT)  # out of the map
    assert (s2, r, done, info['bump']) == (s, -1.0, False, True)
    s2, r, done, _, info = env.step(RIGHT)
    assert (r, done, info['bump']) == (0.0, False, False)
    env.reset(g.idx((1, 2)))
    _, r, done, _, _ = env.step(LEFT)
    assert (r, done) == (20.0, True)


def test_reward_noise_has_paper_means():
    g = Grid.load('ttg')
    env = SingleRobotEnv(g, (0, 4), (0, 2), RewardConfig(noise_sigma=1.0), seed=0)
    samples = []
    for _ in range(2000):
        env.reset()
        samples.append(env.step(LEFT)[1])
    assert abs(np.mean(samples) + 1.0) < 0.1 and 0.9 < np.std(samples) < 1.1


def test_time_limit_truncates():
    g = Grid.load('ttg')
    env = SingleRobotEnv(g, (0, 4), (0, 2), max_steps=3)
    env.reset()
    results = [env.step(LEFT) for _ in range(3)]
    assert [r[3] for r in results] == [False, False, True]


@pytest.mark.parametrize('name', sorted(PAPER_FREE_CELLS))
def test_optimal_q_follows_shortest_path(name):
    g = Grid.load(name)
    for start, goal in g.agents:
        env = SingleRobotEnv(g, start, goal)
        path, reached = greedy_rollout(optimal_q(env), env)
        assert reached and len(path) - 1 == g.distances_to(goal)[g.idx(start)]


def test_q_learning_finds_optimal_path_with_paper_settings():
    g = Grid.load('ttg')
    env = SingleRobotEnv(g, (0, 4), (0, 2))
    res = train(env, episodes=3000, alpha=0.05, epsilon=0.1, seed=1)
    path, reached = greedy_rollout(res.q, env)
    assert reached and len(path) - 1 == 10
    assert policy_agreement(res.q, optimal_q(env), env.goal) > 0.8
    # Reward model: the goal move from (1, 2) always pays +20 when noise is off.
    s, a = g.idx((1, 2)), LEFT
    assert res.reward_count[s, a] > 0 and res.reward_mean[s, a] == 20.0
    assert DELTAS[a] == (-1, 0)
