# Goal-conditioned PPO (Step 4)

This package contains the previous warehouse files plus **new** files:
- `warehouse/goal_env.py`: random start/goal training, four discrete actions, local occupancy input
- `scripts/train_goal_ppo.py`: training script, new independent model
- `scripts/evaluate_goal_ppo.py`: randomized pair evaluation plus mission test
- `scripts/replay_goal_mission.py`: ONE learned policy for all three legs; station waits handled automatically

## Commands

```bash
conda activate warehouse_rl
python -m pip install -r requirements.txt
python -m scripts.train_goal_ppo --timesteps 300000
python -m scripts.evaluate_goal_ppo --episodes 100
python -m scripts.replay_goal_mission --mode check
python -m scripts.replay_goal_mission --mode 3d
```

Training is stochastic. 300k timesteps is a starting budget, **not a guarantee** of reliable mission completion. If evaluation is weak, increase training budget / review rollout traces. No A* override is used in goal-conditioned mission replay; A* distance is only used for training reward shaping and evaluation baselines. `MissionEnv` still enforces 5 simulated seconds of stationary service at each station. In the replay, waits are presented in real time.

The old PPO mission model remains separate. Do not load it with `GoalNavigationEnv` (observation/action spaces differ).
