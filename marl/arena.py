"""N robots moving simultaneously on a paper map, with the team's collision rules.

Each step, every active robot picks one of four moves. Resolution order:

1. Failed move: with probability p_fail the robot stays put (no penalty).
2. Wall: a move into a wall or off the map leaves the robot in place, reward -1.
3. Robot collisions, repeated until nothing changes (a robot sent back can block another):
   - swap: two moving robots trade cells -> both stay, both get -10;
   - same cell: two or more moving robots target one cell -> all stay, all get -10;
   - occupied: a moving robot targets a cell whose robot stays put -> the mover stays and
     gets -10; the robot that stood still is not penalised.
   A robot gets the -10 at most once per step. Only robots that tried to move are penalised.
4. Following: moving into a cell that another robot leaves in the same step is allowed.
5. Goal: a robot that ends the step on its goal gets +20, is done, and leaves the grid,
   so it can no longer collide or block. The episode ends when every robot is done.

With one robot this reduces exactly to SingleRobotEnv.
"""
import numpy as np

from .single_env import RewardConfig

SWAP, SAME_CELL, OCCUPIED = 'swap', 'same_cell', 'occupied'


class MultiRobotEnv:
    def __init__(self, grid, starts=None, goals=None, rewards=RewardConfig(), p_fail=0.0,
                 max_steps=500, seed=None):
        pairs = grid.agents if starts is None else list(zip(starts, goals))
        self.grid = grid
        self.n = len(pairs)
        self.starts = np.array([grid.idx(s) for s, _ in pairs])
        self.goals = np.array([grid.idx(g) for _, g in pairs])
        if len(set(self.starts.tolist())) != self.n:
            raise ValueError('robots must start on distinct cells')
        self.rewards = rewards
        self.p_fail = p_fail
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.pos = self.starts.copy()
        self.done = np.zeros(self.n, dtype=bool)
        self.steps = 0
        return self.pos.copy()

    def step(self, actions):
        g, R = self.grid, self.rewards
        active = np.flatnonzero(~self.done)
        reward = np.zeros(self.n)
        failed = np.zeros(self.n, dtype=bool)
        bumped = np.zeros(self.n, dtype=bool)
        collided = np.zeros(self.n, dtype=bool)
        conflicts = []

        target = self.pos.copy()
        for i in active:
            a = int(actions[i])
            if self.p_fail > 0 and self.rng.random() < self.p_fail:
                failed[i] = True
            elif g.bumps[self.pos[i], a]:
                bumped[i] = True
                reward[i] += R.wall
            else:
                target[i] = g.next_state[self.pos[i], a]

        moving = {int(i) for i in active if target[i] != self.pos[i]}

        def send_back(i, j, kind):
            target[i] = self.pos[i]
            moving.discard(i)
            collided[i] = True
            conflicts.append((min(i, j), max(i, j), kind) if kind != OCCUPIED else (i, j, kind))

        changed = True
        while changed:
            changed = False
            movers = sorted(moving)
            for x, i in enumerate(movers):  # swaps
                for j in movers[x + 1:]:
                    if i in moving and j in moving and target[i] == self.pos[j] and target[j] == self.pos[i]:
                        send_back(i, j, SWAP)
                        send_back(j, i, SWAP)
                        changed = True
            by_cell = {}
            for i in moving:  # several movers into one cell
                by_cell.setdefault(int(target[i]), []).append(i)
            for group in by_cell.values():
                if len(group) > 1:
                    for i in group:
                        send_back(i, next(j for j in group if j != i), SAME_CELL)
                    changed = True
            stays = {int(self.pos[j]): int(j) for j in active if j not in moving}
            for i in sorted(moving):  # mover into a robot that stays put
                if int(target[i]) in stays:
                    send_back(i, stays[int(target[i])], OCCUPIED)
                    changed = True
        # A pair conflict is recorded once per robot; keep one entry per (i, j, kind).
        conflicts = sorted(set(conflicts))

        reward[collided] += R.agent
        self.pos = target
        reached = np.zeros(self.n, dtype=bool)
        for i in active:
            if self.pos[i] == self.goals[i]:
                reached[i] = True
                reward[i] += R.goal
        self.done |= reached
        if R.noise_sigma > 0:
            reward[active] += self.rng.normal(0.0, R.noise_sigma, size=len(active))
        self.steps += 1
        all_done = bool(self.done.all())
        truncated = not all_done and self.steps >= self.max_steps
        info = dict(active=active, failed=failed, bumped=bumped, collided=collided,
                    reached=reached, conflicts=conflicts)
        return self.pos.copy(), reward, self.done.copy(), truncated, info

    def render(self):
        marks = {self.grid.cell(int(gl)): chr(ord('a') + i) for i, gl in enumerate(self.goals)}
        for i in np.flatnonzero(~self.done):
            marks[self.grid.cell(int(self.pos[i]))] = str(i + 1)
        return self.grid.render(marks)
