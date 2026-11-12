"""Run the same A* route in 2D or 3D."""
import argparse
from warehouse.layout import LAYOUT
from warehouse.planner import astar, path_actions
from warehouse.env import WarehouseEnv

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('2d', '3d', 'check'), default='2d')
args = parser.parse_args()

path = astar(LAYOUT)
env = WarehouseEnv()
obs, info = env.reset(seed=42)
terminated = truncated = False
for action in path_actions(path):
    obs, reward, terminated, truncated, info = env.step(action)
    assert not info['collision'], 'Planner hit a shelf'
assert terminated and info['success'], 'Route did not reach the goal'
print(f'Warehouse: {LAYOUT.width}x{LAYOUT.height}; shelf cells: {len(LAYOUT.shelves)}')
print(f'A* route: {len(path)-1} steps; success: {info["success"]}; final position: {info["position"]}')
if args.mode == '2d':
    from warehouse.view2d import play
    play(path)
elif args.mode == '3d':
    from warehouse.view3d import play
    play(path)
else:
    print('Environment verification passed.')
