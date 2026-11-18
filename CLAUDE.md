# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

WPI CS 551 course project with three team members. The proposal (`../RL Proposal.pdf`) defines the **target direction**, which the existing code does not implement yet.

### Target (from the proposal): CQ-learning for multi-robot warehouse coordination
Baseline paper: De Hauwere, Vrancx & Nowé, "Learning multi-agent state space representations", AAMAS 2010 (CQ-learning). Follow-up: FCQ-learning (ALA 2011).
- Each robot k has a **tabular** local Q-table `Q_k(s_k, a_k)` learned first in isolation. It uses that table everywhere except in **coordination states**. A statistical test flags these states when observed rewards differ from the agent's single-agent expectations.
- In coordination states the robot uses a joint-state table `Q^j_k(js, a_k)`, updated with target `r + γ max_a' Q_k(s'_k, a')`. The target bootstraps from the **local** table at the next local state. Each agent still picks only its own action.
- Actions: 4 moves (up/down/left/right, no WAIT). Rewards follow the paper: **+20 goal, −1 wall collision, −10 collision with another agent**. A conflict means two robots entering the same cell or swapping cells. Moves fail with probability p ∈ {0, 0.1, 0.2}. p = 0 reproduces the paper's deterministic setting.
- Objectives:
  - **O1:** replicate CQ-learning on the paper's original grid environments, two agents.
  - **O2:** port it to a narrow-aisle warehouse.
  - **O3:** compare against independent Q-learning (lower bound) and full joint-state Q-learning (reference). Metrics: success rate, steps, collisions, number and location of coordination states (heatmaps), Q-table size. Use 10 seeds.
  - **O4:** scale to 3–4 robots, expanding each coordination state only with the robots involved in the interaction, not all N.
- Stretch goals: DQN function approximation, FCQ-learning, Nash-Q/Correlated-Q in coordination states, dynamic obstacles.
- Deliverables: code, replication results, figures, final report, **2D replay demo**.

### What the paper actually specifies (`../Learning_multi-agent_state_space_representations.pdf`)
- **Prior:** each agent already has an *optimal* single-agent Q_k and a model of expected immediate reward for each (s, a), both learned alone.
- **Detection (Alg. 1, lines 9–17):**
  1. A Student's t-test at **99%** compares the observed immediate rewards for the selected (s_k, a_k) against the expected rewards.
  2. If they differ, the agent collects joint-state samples at s_k.
  3. A second t-test, independent state vs. joint state, marks each js as *dangerous*, which adds it to Q^j_k, or as *safe*.
  4. Coordination is conditioned on the agent's own selected (preferred) action.
- **Learning:**
  - Q^j_k starts empty.
  - Update: `Q^j(js,a) ← (1−α)Q^j(js,a) + α[r + γ max_a' Q_k(s'_k, a')]`, which bootstraps from the local table.
  - **No updates at all in unmarked states**, so Q_k is frozen during the multi-agent phase.
- **Setup:**
  - 2 agents, 4 actions, deterministic transitions.
  - Rewards: +20 goal, −1 wall, −10 agent collision.
  - **For CQ only, rewards are drawn from a normal distribution with these means**; the paper doesn't give the σ.
  - α = 0.05 and fixed ε-greedy ε = 0.1 for all methods.
  - Train 10,000 "iterations", then play greedily for 1,000 and average over 10 runs.
  - γ is not stated.
- **Baselines:**
  - Indep: independent Q-learners.
  - JS: joint-state learners with independent actions.
  - JSA: a joint-state-action "superagent" with 16 actions; it is rewarded when both agents reach their goals, and an agent that reaches its goal stays there.
- **Environments (Fig. 5)**, with free-cell counts = Indep #states, useful for checking map transcriptions:
  - TunnelToGoal (TTG): 25 cells; *both agents share one goal*
  - Two Robot (TR): 36
  - ISR: 43
  - CIT: 69
  - CMU: 133
- **Main result (Table 1):** every method reaches 0 collisions. CQ uses about 1.05–1.1× the Indep state count and needs the fewest steps, e.g. TTG: CQ 27 states / 10.6 steps vs. JS 625 / 13.3.
- **Section 4** (not in our proposal) generalises dangerous/safe samples with a small neural network on (Δx, Δy, a1, a2).
- **Not specified by the paper; we must choose and document:**
  - collision semantics and post-collision positions
  - goal behaviour for Indep, JS and CQ
  - t-test windows and sample counts
  - reward σ
  - γ
  - what an "iteration" is
  - Q^j initialisation

### Who does the work
The teammates are currently inactive, so all work (arena, base Q-learning, CQ) is done here with the user. Planned layout: a new `marl/` package next to `warehouse/`, holding N robots, conflict rules, map files and an N-robot viewer. Training will run on WPI's Academic Turing cluster.

### Current code (preliminary work)
Single-robot only, on the 16×12 grid (24 shelves, 168 free cells). As of commit `c8c4c32`, the main pipeline is **tabular Q-learning** (`warehouse/qlearning.py`); PPO has been dropped from README and `requirements.txt`. The robot navigates start → pickup (wait 5) → drop-off (wait 5) → final goal. Not yet built: the multi-robot arena, failed moves, robot–robot conflicts, the proposal's reward scheme, and anything CQ-specific.

## Commands

All scripts are run as modules **from the repo root** (they import `warehouse.*` and `scripts.*` as packages):

```bash
python -m pip install -r requirements.txt          # Python 3.11; NumPy-only training, CPU

# A* sanity baselines (no display needed)
python -m scripts.demo --mode check                 # fixed start/goal, 22 moves
python -m scripts.mission_demo --mode check         # full mission, 36 moves + 10 waits = 46 steps

# Tabular Q-learning: --task fixed | goal | mission
python -m scripts.train_qlearning --task goal [--episodes 100000] [--seed 42] [--alpha 0.25] [--gamma 0.99] [--epsilon-end 0.05] [--output PATH]
python -m scripts.evaluate_qlearning --task goal [--episodes 100] [--model outputs/q_goal.npz]
python -m scripts.replay_qlearning --task goal --mode check|2d|3d [--max-steps 220]
```

`--mode 2d` (Pygame) and `--mode 3d` (PyBullet) need a graphical display; use `--mode check` headless. There is no test suite or linter; the `--mode check` runs and `evaluate_qlearning` are the de facto tests. Models are saved as `outputs/q_<task>.npz` (`outputs/` is git-ignored).

The PPO scripts (`*_ppo.py`, `*_goal_ppo.py`, `train_mission.py`, `replay_mission.py`, `replay_goal_mission.py`) are still in `scripts/`, but `stable-baselines3`/`tensorboard` were removed from `requirements.txt`, so they fail on a fresh install. Treat them as legacy.

On this machine `python` is not on PATH. Use `/usr/bin/python3` (3.12; NumPy, SciPy, pytest and Pillow are installed system-wide). It is enough for `marl/`, but not for the Gymnasium/Pygame code in `warehouse/`.

### CQ-learning work (`marl/`)
```bash
python3 -m pytest tests -q                                   # map counts, walls, rewards, Q*, learning
python3 -m scripts.train_single_agent                        # prior Q_k: 5 maps x 2 robots x 10 seeds, ~40 s on 12 cores
python3 -m scripts.train_single_agent --maps ttg --seeds 3 --q-init 0 --start-mode random --noise-sigma 1.0
python3 -m marl.maps.extract_fig5 page-6.png                 # re-derive maps from the paper figure (needs Pillow)
```
- `marl/maps/*.json` holds the paper's five maps: grid rows top to bottom, thin `walls` between cell pairs (only TTG has them), and the two agents' `start`/`goal` in (x, y) with y up. `Grid.load` checks free-cell counts against Table 1.
- `marl/grid.py`: `Grid` indexes local states over free cells only. It precomputes `next_state[s, a]` and `bumps[s, a]` tables, and `distances_to(goal)` is a BFS that respects walls.
- `marl/single_env.py`: `RewardConfig` (+20 / −1 / −10, plus optional Gaussian `noise_sigma`) and `SingleRobotEnv`. A wall bump leaves the robot in place; reaching the goal is terminal.
- `marl/qlearning.py`:
  - `train()` runs tabular Q-learning with fixed ε. Truncation still bootstraps; only reaching the goal is terminal.
  - It also records Welford reward stats (`reward_count/mean/m2`) per (s, a), which are CQ's expected-reward model.
  - `optimal_q()` computes the exact Q* by value iteration, and `policy_agreement()` compares a learned table against it.
- Output: `outputs/single_agent/<map>/agent<k>_seed<s>.npz`, containing `q`, `q_star`, the reward stats, episode curves, `free_cells` and a `meta` JSON with the config and metrics. A summary goes to `outputs/single_agent/runs.csv`.
- Prior-training choices the paper leaves open: γ = 0.95, max 500 steps per episode, fixed starts, optimistic `q_init` = 20.
  - Without optimism, 5 of 20 CMU runs lock onto a detour at α = 0.05.
  - With it, all 100 runs follow the shortest path. Off-path cells are less accurate (policy agreement with Q* is 0.43–0.87).
- `marl/arena.py`: `MultiRobotEnv`, N robots moving simultaneously. **The collision rules were decided with the user:**
  - Swap or same-cell: everyone involved stays put and gets −10.
  - Moving into a robot that stays put: only the mover gets −10.
  - Following into a cell being vacated is allowed.
  - Failed moves (`p_fail`) leave the robot in place with no penalty.
  - A robot that reaches its goal gets +20 and leaves the grid; the episode ends when all robots are done.
  - Resolution repeats to a fixed point, so cascades are handled. With N = 1 the arena equals `SingleRobotEnv` (tested).
- `marl/baselines.py`:
  - `IndepLearner` (own cell) and `JointStateLearner` (Q over own cell × the other robot's cell; a finished robot is index `n_cells`), sharing an `act/update` interface.
  - `train()`, `evaluate()` (one greedy episode, since the maps are deterministic) and `run_episode()`. CQ-learning should plug into the same loop.
- `python3 -m scripts.train_baselines` runs Indep + JS × 5 maps × 10 seeds with paper settings and Q₀ = 0. It prints a Table 1-style summary next to the paper's numbers, and writes to `outputs/baselines/`.
- Greedy Indep play can **deadlock**: both robots keep choosing the same cell, collide and stay put, e.g. at TTG's tunnel entrance. A robot that never arrives is counted as `max_steps`.
- Table 1 "#steps" doesn't match single-robot shortest paths directly. ISR's shortest path is 4, but CQ reports 5; TR's is 10, but Indep reports 9. Don't use it to validate the single-agent prior.

## Architecture

- `warehouse/layout.py`: the `LAYOUT` singleton is the single source of truth for grid size, shelves, start and final goal. Every env, planner and viewer reads it. `blocked()` covers both out-of-bounds cells and shelves.
- `warehouse/env.py`: defines `DELTAS`, the action order **0:right, 1:up, 2:left, 3:down**. y increases upward, and viewers flip y when drawing. It also has the fixed start/goal `WarehouseEnv`, rewarded by Manhattan progress.
- `warehouse/goal_env.py`: `GoalNavigationEnv` samples random reachable start/goal pairs (BFS distance from 3 to `max_steps - 5`), or accepts `reset(options={'start':..., 'goal':...})`. Reward is −0.1 per step, +1 per unit of BFS-distance progress, −2 on a wall bump and +50 at the goal. BFS distance maps are cached per target in `_dist`.
- `warehouse/mission.py`: `MissionEnv` has 5 actions, with action 4 = `WAIT`. It holds the station constants (`PICKUP`, `DROPOFF`, `STATIONS`) and the stage machine (PICKUP → DROPOFF → FINAL GOAL → COMPLETE). On a pickup/drop-off station the env forces `WAIT` until 5 waits complete. The file also has `shortest_path` (BFS) and `baseline_actions`/`rollout` for the A* demo.
- `warehouse/qlearning.py`:
  - `QTable(kind)` stores a dense NumPy table indexed by raw `(x, y)` coordinates, blocked cells included. The shape depends on `kind`:
    - `fixed`: `[x, y, a]`
    - `goal`: `[x, y, gx, gy, a]`, a single table for all goals; `q[:, :, gx, gy, :]` is the local table for one goal
    - `mission`: `[stage, x, y, wait, a]` with 5 actions
  - `QTable.state(env)` reads `env.robot` / `env.goal` / `env.stage` directly, not the Gym observation vector.
  - `train()` runs an epsilon-greedy loop: epsilon decays linearly to `epsilon_end` over 80% of the episodes, and episode i is reset with `seed + i`.
  - Truncated transitions are treated as terminal and bootstrap zero.
  - The `.npz` file stores `q`, `kind`, `alpha` and `gamma` only.
- `scripts/replay_qlearning.py::run_mission`: for a `goal` table it steps `MissionEnv(final_goal_assist=False)`. On each non-wait step it builds the state from a `GoalNavigationEnv` reset to `(mission.robot, mission.target)`, so one goal-conditioned table drives all three legs. `evaluate_qlearning` reuses it.
- `warehouse/mission_view.py`: `play2d`/`play3d` replay frame dicts (`MissionEnv` info plus `action`). `view2d.py`/`view3d.py` are route-only viewers. `warehouse/planner.py` is A*, used only for baselines.

## Gaps between current Q-learning and what CQ-learning needs

- The current rewards are the shaped goal-env rewards. Local Q-tables for CQ must be trained in the multi-robot arena with N = 1, using the proposal's rewards (+20 / −1 / −10, no shaping) and the same `p_fail`. Otherwise the reward-based coordination-state test fires on reward-scheme differences instead of real interactions.
- CQ's statistical test needs per-(state, action) immediate-reward samples from single-agent learning. `QTable.save` doesn't record any.
- The tables are indexed by raw x/y coordinates, while the arena plan uses free-cell indices. Converting between them is a trivial reshape and mask, but agree on one convention before the handoff.

## Pitfalls / conventions

- `MissionEnv(final_goal_assist=True)` is the **constructor default** and makes A* drive the whole final leg. The Q-learning scripts pass `False`. Any result with assist on is a hybrid controller and must not be reported as learned-only. Never add undocumented A* overrides to mask a failing policy.
- BFS/A* may be used for reward shaping and benchmarking, but never for choosing learned actions.
- Results are tied to the single fixed shelf layout. Random goals do not mean randomized geometry.
