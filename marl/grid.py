"""Grid maps with blocked cells and thin walls between cells.

Coordinates are (x, y) with x to the right and y upward, as in warehouse/layout.py.
Map files store rows top to bottom, so the first text row has y = height - 1.
Local states are indices over free cells only (0 .. n_free - 1).
"""
from collections import deque
import json
from pathlib import Path

import numpy as np

# Same action order as warehouse/env.py: 0 right, 1 up, 2 left, 3 down.
DELTAS = ((1, 0), (0, 1), (-1, 0), (0, -1))
ACTION_NAMES = ('right', 'up', 'left', 'down')
MAP_DIR = Path(__file__).resolve().parent / 'maps'


class Grid:
    def __init__(self, rows, walls=(), name='grid', agents=()):
        self.name = name
        self.height = len(rows)
        self.width = len(rows[0])
        if any(len(r) != self.width for r in rows):
            raise ValueError('map rows must all have the same length')
        self.free = [(x, self.height - 1 - r) for r, row in enumerate(rows)
                     for x, ch in enumerate(row) if ch == '.']
        self.free.sort(key=lambda c: (-c[1], c[0]))  # reading order: top row first
        self._index = {c: i for i, c in enumerate(self.free)}
        self.walls = set()
        for a, b in walls:
            a, b = tuple(a), tuple(b)
            if a not in self._index or b not in self._index or abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                raise ValueError(f'wall {a}-{b} is not between two adjacent free cells')
            self.walls.add(frozenset((a, b)))
        self.agents = [(tuple(a['start']), tuple(a['goal'])) for a in agents]
        # next_state[s, a] is the cell index after action a; bumps[s, a] is True if blocked.
        n = len(self.free)
        self.next_state = np.empty((n, 4), dtype=np.int64)
        self.bumps = np.zeros((n, 4), dtype=bool)
        for i, (x, y) in enumerate(self.free):
            for a, (dx, dy) in enumerate(DELTAS):
                nxt = (x + dx, y + dy)
                if nxt in self._index and frozenset(((x, y), nxt)) not in self.walls:
                    self.next_state[i, a] = self._index[nxt]
                else:
                    self.next_state[i, a] = i
                    self.bumps[i, a] = True

    @classmethod
    def load(cls, name_or_path):
        path = Path(name_or_path)
        if not path.suffix:
            path = MAP_DIR / f'{name_or_path}.json'
        doc = json.loads(path.read_text())
        grid = cls(doc['grid'], doc.get('walls', ()), doc.get('name', path.stem), doc.get('agents', ()))
        expected = doc.get('expected_free_cells')
        if expected is not None and grid.n_states != expected:
            raise ValueError(f'{path.name}: {grid.n_states} free cells, expected {expected}')
        return grid

    @property
    def n_states(self):
        return len(self.free)

    def idx(self, cell):
        return self._index[tuple(cell)]

    def cell(self, idx):
        return self.free[idx]

    def distances_to(self, goal):
        """BFS shortest-path length from every free cell to `goal` (walls respected)."""
        g = self.idx(goal)
        dist = np.full(self.n_states, -1, dtype=np.int64)
        dist[g] = 0
        queue = deque([g])
        while queue:
            s = queue.popleft()
            for a in range(4):
                # Moves are symmetric, so neighbours of s are cells that reach s in one move.
                p = self.next_state[s, a]
                if dist[p] < 0 and not self.bumps[s, a]:
                    dist[p] = dist[s] + 1
                    queue.append(p)
        return dist

    def render(self, marks=None):
        """ASCII picture; `marks` maps (x, y) -> single character."""
        marks = marks or {}
        lines = []
        for y in range(self.height - 1, -1, -1):
            row = ''
            for x in range(self.width):
                row += marks.get((x, y), '.' if (x, y) in self._index else '#')
            lines.append(row)
        return '\n'.join(lines)


PAPER_MAPS = ['ttg', 'tr', 'isr', 'cit', 'cmu']  # Figure 5 of De Hauwere et al. (2010)


def available_maps():
    return sorted(p.stem for p in MAP_DIR.glob('*.json'))
