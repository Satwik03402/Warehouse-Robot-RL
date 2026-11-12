# Warehouse Robot — Tabular Q-Learning

This is the **Q-learning** version of the warehouse robot project. It keeps the 16×12 map,
24 shelf cells, Gymnasium environments, station service/waits, and Pygame/PyBullet
visualizations. There is **no PPO, neural network, or Stable-Baselines3 dependency**.

## Install and run (from project root)

```bash
pip install -r requirements.txt
python -m scripts.train_qlearning --task goal --episodes 100000
python -m scripts.evaluate_qlearning --task goal --episodes 100
python -m scripts.replay_qlearning --task goal --mode 3d
```

Alternative experiments:

```bash
# Fixed start and target: fast sanity check
python -m scripts.train_qlearning --task fixed --episodes 15000
python -m scripts.evaluate_qlearning --task fixed
python -m scripts.replay_qlearning --task fixed --mode 2d

# Train Q-learning directly on the full stage-aware mission, including WAIT
python -m scripts.train_qlearning --task mission --episodes 40000
python -m scripts.evaluate_qlearning --task mission
python -m scripts.replay_qlearning --task mission --mode 3d
```

Use `--mode check` for a headless run and `--mode 2d` for Pygame.
Models are `.npz` files saved in `outputs/`, with one learned Q value for each
state/action combination. Training uses CPU and NumPy only.

## What is actually learned?

- **Fixed:** Q[x, y, action], with 4 navigation actions.
- **Goal:** Q[x, y, goal_x, goal_y, action] for random start/goal pairs;
  the **same learned table** controls pickup, drop-off, and final-goal movement.
  Waiting at stations is handled automatically by the mission environment.
- **Mission:** Q[stage, x, y, wait_progress, action], with a fifth WAIT action.
  A full-stage controller learns all actions rather than using a separate navigator.

Each training update is:

`Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]`

Terminal/truncated transitions bootstrap zero. Epsilon-greedy exploration decays
linearly from 1.0 to 0.05 (by 80% of the total episodes). Defaults are
`alpha=0.25`, `gamma=0.99`. Set `--episodes`, `--alpha`, `--gamma`, and
`--epsilon-end` to tune training.

The goal environment's existing shortest-path-distance reward shaping is
retained, and distances are now cached for training performance. This means
training uses a map-aware shaping reward, **but the Q-learning action selection
at replay does not call A***. For an entirely map-agnostic learning experiment,
replace distance-based shaping with a simpler reward and retrain.

## Evaluate carefully

The fixed and mission tasks use a single deterministic start/layout/route, so
repeating their evaluation does not demonstrate generalization. The randomized
`goal` environment tests unseen sampled start/target pairs; not every possible
pair is guaranteed well trained, and success depends on training episodes.

`python -m scripts.demo --mode check` and `python -m scripts.mission_demo --mode check`
remain as **A* validation baselines**, not Q-learning controllers.

Mission service time is **five WAIT actions at each of two stations**.
`MissionEnv(final_goal_assist=False)` is used during Q-learning training and
replay, so there is **no A* final-leg action override**.
