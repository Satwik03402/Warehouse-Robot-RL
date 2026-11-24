import numpy as np
import pytest

from marl.arena import OCCUPIED, SAME_CELL, SWAP, MultiRobotEnv
from marl.grid import Grid
from marl.single_env import RewardConfig, SingleRobotEnv

RIGHT, UP, LEFT, DOWN = range(4)
CORRIDOR = Grid(['......'])                 # one row, x = 0..5, y = 0
ROOM = Grid(['...', '...', '...'])          # 3 x 3, y = 0 is the bottom row


def make(grid, starts, goals, **kw):
    return MultiRobotEnv(grid, starts=starts, goals=goals, **kw)


def cells(env):
    return [env.grid.cell(int(s)) for s in env.pos]


def test_swap_both_stay_and_both_penalised():
    env = make(CORRIDOR, [(1, 0), (2, 0)], [(5, 0), (0, 0)])
    _, r, _, _, info = env.step([RIGHT, LEFT])
    assert cells(env) == [(1, 0), (2, 0)]
    assert r.tolist() == [-10, -10]
    assert info['conflicts'] == [(0, 1, SWAP)]


def test_same_cell_both_stay_and_both_penalised():
    env = make(CORRIDOR, [(1, 0), (3, 0)], [(5, 0), (0, 0)])
    _, r, _, _, info = env.step([RIGHT, LEFT])
    assert cells(env) == [(1, 0), (3, 0)]
    assert r.tolist() == [-10, -10]
    assert info['conflicts'] == [(0, 1, SAME_CELL)]


def test_three_robots_into_one_cell():
    env = make(ROOM, [(0, 1), (2, 1), (1, 0)], [(2, 2), (0, 2), (1, 2)])
    _, r, _, _, _ = env.step([RIGHT, LEFT, UP])
    assert cells(env) == [(0, 1), (2, 1), (1, 0)]
    assert r.tolist() == [-10, -10, -10]


def test_moving_into_stationary_robot_penalises_only_the_mover():
    env = make(CORRIDOR, [(1, 0), (2, 0)], [(5, 0), (0, 0)])
    # In a one-row corridor UP always bumps the edge, so robot 2 stays put.
    _, r, _, _, info = env.step([RIGHT, UP])
    assert cells(env) == [(1, 0), (2, 0)]
    assert r.tolist() == [-10, -1]          # robot 2 only pays for its own wall bump
    assert info['conflicts'] == [(0, 1, OCCUPIED)]
    assert info['collided'].tolist() == [True, False]


def test_following_is_allowed():
    env = make(CORRIDOR, [(1, 0), (2, 0)], [(5, 0), (4, 0)])
    _, r, _, _, info = env.step([RIGHT, RIGHT])
    assert cells(env) == [(2, 0), (3, 0)]
    assert r.tolist() == [0, 0] and info['conflicts'] == []


def test_rotation_of_three_is_allowed():
    env = make(ROOM, [(0, 0), (1, 0), (1, 1)], [(2, 2), (0, 2), (2, 0)])
    # (0,0) -> (1,0) -> (1,1) -> (0,1): a chain where each robot follows the next.
    _, r, _, _, info = env.step([RIGHT, UP, LEFT])
    assert cells(env) == [(1, 0), (1, 1), (0, 1)]
    assert info['conflicts'] == []


def test_cascade_blocked_follower_is_penalised():
    # Robots 1 and 2 swap (both sent back); robot 3 was following robot 2, so it now
    # runs into a robot that stays put.
    env = make(CORRIDOR, [(1, 0), (2, 0), (3, 0)], [(5, 0), (0, 0), (0, 0)])
    _, r, _, _, info = env.step([RIGHT, LEFT, LEFT])
    assert cells(env) == [(1, 0), (2, 0), (3, 0)]
    assert r.tolist() == [-10, -10, -10]
    assert (2, 1, OCCUPIED) in info['conflicts']


def test_failed_move_stays_without_penalty_and_blocks_others():
    env = make(CORRIDOR, [(1, 0), (2, 0)], [(5, 0), (4, 0)], p_fail=1.0)
    _, r, _, _, info = env.step([RIGHT, RIGHT])
    assert cells(env) == [(1, 0), (2, 0)]
    assert r.tolist() == [0, 0] and info['failed'].all()


def test_goal_removes_robot_and_episode_ends_when_all_done():
    env = make(CORRIDOR, [(3, 0), (1, 0)], [(4, 0), (5, 0)])
    _, r, done, _, _ = env.step([RIGHT, RIGHT])
    assert r.tolist() == [20, 0] and done.tolist() == [True, False]
    # Robot 2 can now pass through robot 1's goal cell without colliding.
    env.step([LEFT, RIGHT])
    env.step([LEFT, RIGHT])
    _, r, done, _, info = env.step([LEFT, RIGHT])
    assert r.tolist() == [0, 20] and done.all() and info['conflicts'] == []


def test_shared_goal_ttg_second_robot_can_still_arrive():
    g = Grid.load('ttg')
    env = MultiRobotEnv(g)
    env.pos[:] = [g.idx((1, 2)), g.idx((2, 2))]   # both in the tunnel, robot 1 in front
    _, r, done, _, _ = env.step([LEFT, LEFT])      # robot 1 reaches the goal, robot 2 follows
    assert done.tolist() == [True, False] and r.tolist() == [20, 0]
    _, r, done, _, info = env.step([LEFT, LEFT])   # robot 1 has left the grid
    assert done.all() and r[1] == 20 and info['conflicts'] == []


def test_one_robot_matches_single_robot_env():
    g = Grid.load('cit')
    (start, goal), = g.agents[:1]
    multi = MultiRobotEnv(g, starts=[start], goals=[goal])
    single = SingleRobotEnv(g, start, goal)
    single.reset()
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = int(rng.integers(4))
        s1, r1, d1, t1, _ = single.step(a)
        sm, rm, dm, tm, _ = multi.step([a])
        assert (s1, r1, d1, t1) == (sm[0], rm[0], dm[0], tm)
        if d1:
            break


def test_same_seed_same_trajectory_with_noise_and_failures():
    def run(seed):
        env = MultiRobotEnv(Grid.load('tr'), rewards=RewardConfig(noise_sigma=1.0), p_fail=0.2, seed=seed)
        rng = np.random.default_rng(1)
        return [env.step(rng.integers(4, size=2))[1].tolist() for _ in range(50)]
    assert run(3) == run(3) and run(3) != run(4)


def test_distinct_starts_required():
    with pytest.raises(ValueError):
        make(CORRIDOR, [(1, 0), (1, 0)], [(5, 0), (0, 0)])
