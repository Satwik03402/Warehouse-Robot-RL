"""A* baseline, used to validate reachability; not an RL policy."""
from heapq import heappush, heappop
from .env import DELTAS

def astar(layout):
    start, goal = layout.start, layout.goal
    frontier = [(0, 0, start)]
    parent = {start: None}
    cost = {start: 0}
    index = 0
    while frontier:
        _, _, current = heappop(frontier)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        for dx, dy in DELTAS:
            candidate = (current[0] + dx, current[1] + dy)
            new_cost = cost[current] + 1
            if layout.blocked(candidate) or new_cost >= cost.get(candidate, float('inf')):
                continue
            cost[candidate] = new_cost
            parent[candidate] = current
            index += 1
            h = abs(goal[0] - candidate[0]) + abs(goal[1] - candidate[1])
            heappush(frontier, (new_cost + h, index, candidate))
    raise RuntimeError('No route between warehouse start and goal')

def path_actions(path):
    reverse = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    return [reverse[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:])]
