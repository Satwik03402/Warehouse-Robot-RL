"""Replay grid path in a matching PyBullet warehouse (kinematic visualization)."""
import time
import pybullet as p
import pybullet_data
from .layout import LAYOUT

def play(path, fps=12):
    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError('Cannot start PyBullet GUI (check desktop DISPLAY)')
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF('plane.urdf')
        p.resetDebugVisualizerCamera(cameraDistance=19, cameraYaw=42, cameraPitch=-55,
                                     cameraTargetPosition=[LAYOUT.width / 2, LAYOUT.height / 2, 0])
        def cube(center, half, rgba, mass=0):
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
            return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape,
                                     baseVisualShapeIndex=visual, basePosition=center)
        # One 1-m cube per blocked grid cell, grouped visually into four shelf runs.
        for x, y in sorted(LAYOUT.shelves):
            cube((x + .5, y + .5, .7), (.47, .47, .7), (.39, .46, .55, 1))
        # Outer wall segments.
        cube((LAYOUT.width / 2, -.08, .6), (LAYOUT.width / 2, .08, .6), (.6, .64, .69, 1))
        cube((LAYOUT.width / 2, LAYOUT.height + .08, .6), (LAYOUT.width / 2, .08, .6), (.6, .64, .69, 1))
        cube((-.08, LAYOUT.height / 2, .6), (.08, LAYOUT.height / 2, .6), (.6, .64, .69, 1))
        cube((LAYOUT.width + .08, LAYOUT.height / 2, .6), (.08, LAYOUT.height / 2, .6), (.6, .64, .69, 1))
        def marker(cell, color):
            x, y = cell
            cube((x + .5, y + .5, .04), (.42, .42, .04), color)
        marker(LAYOUT.start, (1, .72, .22, 1))
        marker(LAYOUT.goal, (.12, .78, .32, 1))
        robot = cube((path[0][0] + .5, path[0][1] + .5, .27), (.33, .33, .25), (.14, .35, .9, 1))
        for a, b in zip(path, path[1:]):
            p.addUserDebugLine([a[0] + .5, a[1] + .5, .12],
                               [b[0] + .5, b[1] + .5, .12], [0.06, .72, .94], lineWidth=3)
        print('PyBullet GUI running. Close with Ctrl+C in terminal.')
        for index in range(len(path) - 1):
            a, b = path[index], path[index + 1]
            for t in range(fps):
                alpha = (t + 1) / fps
                x = a[0] + (b[0] - a[0]) * alpha + .5
                y = a[1] + (b[1] - a[1]) * alpha + .5
                p.resetBasePositionAndOrientation(robot, (x, y, .27), (0, 0, 0, 1))
                p.stepSimulation()
                time.sleep(1 / 60)
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()
