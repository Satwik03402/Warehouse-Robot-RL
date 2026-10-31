# Warehouse Robot Navigation with Goal-Conditioned PPO

A Python reinforcement-learning project for an autonomous warehouse mobile robot. The robot navigates to a **pickup station**, waits for **5 simulated seconds** to load a package, navigates to a **drop-off station**, waits another **5 simulated seconds** to unload, and finally reaches its **final goal**.

The learning environment is a lightweight **2D grid warehouse** for fast experimentation. Mission trajectories can be replayed in **2D (Pygame)** or **3D (PyBullet)** using the same warehouse layout.

> **Current scope:** Grid-based navigation with four discrete actions, not continuous differential-drive or physical robot control. Pickup and drop-off are simulated station operations, not robotic grasping. The 3D viewer replays navigation trajectories.

## Demo and results

The main controller uses **one goal-conditioned PPO policy for all three navigation legs**. At the pickup and drop-off stations, an automatic mission manager applies the required waiting period. **No A\* action assistance** is used in the goal-conditioned PPO mission replay.

Results from a local run (100 random start-goal evaluation episodes on the **same fixed warehouse layout**):

| Metric | Observed result |
| --- | ---: |
| Navigation success | **99/100 (99%)** |
| Average navigation steps | **11.3** |
| Average collisions | **0.00** |
| Average extra steps on successful runs vs. shortest path | **0.08** |
| Full pickup → drop-off → final-goal mission | **Success** |
| Full mission steps | **46** |
| Full mission collisions | **0** |
| Full mission reward | **183.4** |
| Final goal cell | **(14, 10)** |

These numbers describe one recorded evaluation, **not a guarantee of results on every run**. Random starts and goals test navigation on the known shelf layout; generalization to unseen shelf configurations and moving obstacles is future work.

### Warehouse and mission

- Grid dimensions: **16 × 12**; **24** shelf cells.
- Start: `(1, 1)`
- Pickup: `(2, 9)` — wait **5 simulated seconds**.
- Drop-off: `(13, 2)` — wait **5 simulated seconds**.
- Final goal: `(14, 10)`.
- Navigation actions: up, down, left, right (discrete grid moves).
- The mission manager selects the current target and performs loading/unloading automatically after the required stationary wait.

## Technology

- Python 3.11
- Gymnasium — custom RL environments
- Stable-Baselines3 / PyTorch — PPO
- NumPy — observations and data
- Pygame — 2D visualization
- PyBullet — 3D mission replay
- TensorBoard / Matplotlib — training logs and analysis

Training and inference for the provided PPO scripts use **CPU** by default. An NVIDIA GPU is helpful for other projects but not required here.

## Setup

Run the commands from the **repository root** (the folder containing `warehouse/`, `scripts/`, and `requirements.txt`).

```bash
# Clone after publishing to GitHub
# git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
# cd YOUR_REPO

# Create the Python environment (Conda)
conda create -n warehouse_rl python=3.11 -y
conda activate warehouse_rl

# Install Python dependencies
python -m pip install -r requirements.txt

# Verify the installation
python -c "import gymnasium, pygame, pybullet, stable_baselines3, tensorboard; print('Dependencies OK')"
```

Alternatively, with `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Graphical 2D/3D replay requires a desktop session with a working display; training and `--mode check` do not.

## Quick start

### 1. Verify the warehouse and A* baseline

```bash
python -m scripts.demo --mode check
python -m scripts.demo --mode 2d
python -m scripts.demo --mode 3d
```

This is the original fixed-start, fixed-goal baseline. Its shortest path contains **22 moves**.

### 2. Check the complete pickup/drop-off mission using A*

```bash
python -m scripts.mission_demo --mode check
python -m scripts.mission_demo --mode 2d
python -m scripts.mission_demo --mode 3d
```

This validates station logic independently from learned navigation. The baseline mission consists of **36 movement actions + 10 wait actions = 46 simulated steps**.

### 3. Train goal-conditioned PPO (recommended model)

```bash
python -m scripts.train_goal_ppo --timesteps 300000
```

The model is saved to:

```text
outputs/ppo_goal_navigation.zip
```

Optional different training budget / seed:

```bash
python -m scripts.train_goal_ppo --timesteps 500000 --seed 42
```

PPO learns navigation from randomized reachable start-goal pairs on the existing warehouse layout. Observations include robot position, goal position, relative goal displacement, blocked neighboring cells, and a local occupancy patch. Rewards use progress in shortest-path distance around shelves; **A* is not used to choose PPO actions**.

### 4. Evaluate the trained navigation policy

```bash
python -m scripts.evaluate_goal_ppo --episodes 100
```

To evaluate a different model file:

```bash
python -m scripts.evaluate_goal_ppo --episodes 100 --model outputs/ppo_goal_navigation.zip
```

The script reports random-pair success rate, steps, collisions, path-length excess versus optimal routes, and the full three-leg mission result.

### 5. Replay the entire PPO mission

```bash
# Text-only verification
python -m scripts.replay_goal_mission --mode check

# 2D animated replay
python -m scripts.replay_goal_mission --mode 2d

# 3D PyBullet animated replay
python -m scripts.replay_goal_mission --mode 3d
```

Optional model path and maximum steps:

```bash
python -m scripts.replay_goal_mission --mode 3d \
  --model outputs/ppo_goal_navigation.zip \
  --max-steps 220
```

**Expected behavior:** navigate to pickup → wait 5 seconds → carry package to drop-off → wait 5 seconds → reach final goal. The waits represent five simulated one-second actions, shown in real time during mission replay; training is not slowed by wall-clock waiting.

## Earlier experiments (optional)

For comparison, the repository also includes earlier fixed-goal and mission-level controllers:

```bash
# Fixed start/goal PPO
python -m scripts.train_ppo --timesteps 200000
python -m scripts.evaluate_ppo --episodes 30
python -m scripts.replay_ppo --mode 2d
python -m scripts.replay_ppo --mode 3d

# Earlier stage-based mission PPO
python -m scripts.train_mission --timesteps 500000
python -m scripts.replay_mission --mode check
python -m scripts.replay_mission --mode 3d

# Unassisted mission check (disables classical final-leg assistance)
python -m scripts.replay_mission --mode check --no-final-assist
```

**Important:** The earlier hybrid mission may use A* for the final leg. Do **not** label those results "PPO-only." The recommended `replay_goal_mission` command uses goal-conditioned PPO for all navigation legs without A* action assistance.

## Project structure

```text
.
├── warehouse/
│   ├── layout.py             # Single shared map and obstacles
│   ├── env.py                # Original warehouse navigation environment
│   ├── goal_env.py           # Random-start/goal PPO training environment
│   ├── mission.py            # Pickup, drop-off, waits, completion logic
│   ├── planner.py            # A* baseline utilities
│   ├── mission_view.py       # 2D / 3D mission visualization
│   ├── view2d.py             # Original 2D route viewer
│   └── view3d.py             # Original 3D route viewer
├── scripts/
│   ├── demo.py
│   ├── mission_demo.py
│   ├── train_goal_ppo.py
│   ├── evaluate_goal_ppo.py
│   ├── replay_goal_mission.py
│   ├── train_ppo.py
│   ├── evaluate_ppo.py
│   ├── replay_ppo.py
│   ├── train_mission.py
│   └── replay_mission.py
├── requirements.txt
├── README.md
└── outputs/                   # Generated models/logs; ignored by Git
```

## Training and evaluation notes

- Training is stochastic. Your model's results may differ across seeds and budgets.
- Goal-conditioned PPO currently trains with the **same fixed shelf layout**. Randomizing goals is different from randomizing the environment geometry.
- Shortest-path distances are used for reward shaping and benchmarking, but PPO chooses its own actions during the recommended mission replay.
- Pickup/drop-off waits are controlled by mission logic, **not learned WAIT decisions**.
- Compare PPO, A*, and hybrid approaches fairly by using the same start-goal pairs and reporting success rate, collisions, and path length.
- Large policy checkpoints and logs are excluded from version control by default. Run the training command to regenerate a checkpoint, or publish a small vetted model separately as a GitHub Release if desired.

## Troubleshooting

**`ImportError: ... tensorboard is not installed`**

```bash
python -m pip install tensorboard
```

**`FileNotFoundError` / missing `ppo_goal_navigation.zip`**

```bash
python -m scripts.train_goal_ppo --timesteps 300000
```

**No graphical window or X11 error on Ubuntu**

Use `--mode check` to test the simulator without graphics. Run PyBullet/Pygame from an active graphical session; an `X connection ... broken` message after you close the viewer may simply indicate the display connection was shut down.

**Poor PPO performance / oscillation near a target**

Evaluate random start-goal episodes first; inspect path-length excess and collisions. Avoid masking a failed learned policy with an undocumented A* override. For a live demo, you may separately present the hybrid controller and identify it as such.

## Next steps

- Evaluate multiple seeds and hold-out start-goal combinations.
- Train/test on randomized shelf layouts.
- Add moving obstacles and dynamic replanning experiments.
- Add multi-robot traffic and warehouse task allocation.

## Acknowledgments

Built with Gymnasium, Stable-Baselines3, Pygame, and PyBullet.
