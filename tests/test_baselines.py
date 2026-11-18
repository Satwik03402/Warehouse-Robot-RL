import numpy as np

from marl.arena import MultiRobotEnv
from marl.baselines import IndepLearner, JointStateLearner, evaluate, train
from marl.grid import Grid

RIGHT, UP, LEFT, DOWN = range(4)


def test_joint_state_table_shape_and_absent_robot_index():
    learner = JointStateLearner(0, n_cells=25, n_robots=2)
    assert learner.q.shape == (25, 26, 4) and learner.n_states == 625
    assert learner.key(np.array([3, 7]), np.array([False, False])) == (3, 7)
    assert learner.key(np.array([3, 7]), np.array([False, True])) == (3, 25)


def test_update_touches_only_the_taken_joint_state_action():
    learner = JointStateLearner(0, n_cells=25, n_robots=2, alpha=0.5)
    pos, nxt = np.array([3, 7]), np.array([4, 7])
    none = np.array([False, False])
    learner.update(pos, none, RIGHT, -10.0, nxt, none)
    changed = np.argwhere(learner.q != 0)
    assert changed.tolist() == [[3, 7, RIGHT]] and learner.q[3, 7, RIGHT] == -5.0


def test_indep_update_terminal_on_goal():
    learner = IndepLearner(1, n_cells=25, n_robots=2, alpha=1.0, q_init=5.0)
    learner.update(np.array([0, 3]), np.array([False, False]), LEFT, 20.0,
                   np.array([0, 2]), np.array([False, True]))
    assert learner.q[3, LEFT] == 20.0  # no bootstrap after reaching the goal


def test_both_learners_solve_a_corridor_where_robots_never_meet():
    g = Grid(['......', '######', '......'])
    for cls in (IndepLearner, JointStateLearner):
        env = MultiRobotEnv(g, starts=[(0, 2), (0, 0)], goals=[(5, 2), (5, 0)], max_steps=50)
        learners = [cls(i, g.n_states, 2, alpha=0.5) for i in range(2)]
        train(env, learners, episodes=2000, seed=0)  # JS needs many visits per joint state
        ev = evaluate(env, learners)
        assert ev['success'] and ev['steps_per_robot'] == [5, 5] and ev['collisions'] == 0
