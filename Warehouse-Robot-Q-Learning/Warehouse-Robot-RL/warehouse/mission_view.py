"""Synchronized 2D and 3D mission viewers. 1 WAIT action = 1 real-time second."""
import time
from .layout import LAYOUT
from .mission import PICKUP, DROPOFF, WAIT


def play2d(frames, move_seconds=0.22):
    import pygame
    pygame.init()
    size=45; banner=70
    screen=pygame.display.set_mode((LAYOUT.width*size, LAYOUT.height*size+banner))
    pygame.display.set_caption('Warehouse | Pickup -> Drop-off -> Goal')
    font=pygame.font.SysFont(None, 27)
    colors={LAYOUT.start:(245,181,63), PICKUP:(50,194,226), DROPOFF:(205,105,223), LAYOUT.goal:(70,201,114)}
    def cell_rect(c):return pygame.Rect(c[0]*size,(LAYOUT.height-1-c[1])*size,size,size)
    try:
        for frame in frames:
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT or (ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE):return
            screen.fill((239,243,247))
            for x in range(LAYOUT.width):
                for y in range(LAYOUT.height):pygame.draw.rect(screen,(211,220,228),cell_rect((x,y)),1)
            for c in LAYOUT.shelves:pygame.draw.rect(screen,(94,109,128),cell_rect(c).inflate(-3,-3),border_radius=4)
            for c,color in colors.items():pygame.draw.rect(screen,color,cell_rect(c).inflate(-9,-9),border_radius=6)
            pygame.draw.circle(screen,(33,89,205),cell_rect(frame['position']).center,15)
            if frame['carrying']:
                pygame.draw.rect(screen,(255,220,54),cell_rect(frame['position']).inflate(-25,-25))
            pygame.draw.rect(screen,(34,42,59),(0,LAYOUT.height*size,LAYOUT.width*size,banner))
            desc=f"Stage: {frame['stage']}    Wait: {frame['wait_seconds']}/5s    Package: {'YES' if frame['carrying'] else 'NO'}"
            screen.blit(font.render(desc,True,(255,255,255)),(14,LAYOUT.height*size+15))
            screen.blit(font.render('Cyan: pickup   Purple: drop-off   Green: final goal',True,(202,209,220)),(14,LAYOUT.height*size+42))
            pygame.display.flip()
            time.sleep(1.0 if frame['action']==WAIT else move_seconds)
        # Final state stays visible until user exits
        while True:
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT or (ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE):return
            time.sleep(0.03)
    finally:pygame.quit()


def play3d(frames, move_seconds=0.22):
    import pybullet as p
    import pybullet_data
    cid=p.connect(p.GUI)
    if cid<0:raise RuntimeError('Cannot open PyBullet graphical display')
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        p.loadURDF('plane.urdf')
        p.resetDebugVisualizerCamera(cameraDistance=19,cameraYaw=42,cameraPitch=-55,cameraTargetPosition=[8,6,0])
        def block(position, half, color):
            c=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
            v=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=color)
            return p.createMultiBody(baseMass=0,baseCollisionShapeIndex=c,baseVisualShapeIndex=v,basePosition=position)
        for x,y in LAYOUT.shelves:block((x+.5,y+.5,.7),(.47,.47,.7),(.39,.46,.55,1))
        for pos,half in [((8,-.08,.6),(8,.08,.6)),((8,12.08,.6),(8,.08,.6)),((-.08,6,.6),(.08,6,.6)),((16.08,6,.6),(.08,6,.6))]:block(pos,half,(.6,.64,.69,1))
        for c,color in [(LAYOUT.start,(1,.72,.22,1)),(PICKUP,(0,.8,1,1)),(DROPOFF,(.83,.35,.92,1)),(LAYOUT.goal,(.12,.78,.32,1))]:
            block((c[0]+.5,c[1]+.5,.04),(.42,.42,.04),color)
        pos=frames[0]['position']
        robot=block((pos[0]+.5,pos[1]+.5,.27),(.32,.32,.25),(.14,.35,.9,1))
        pkg=block((pos[0]+.5,pos[1]+.5,-1),(.18,.18,.13),(1,.76,.11,1))
        text_id=-1
        for i,frame in enumerate(frames):
            if not p.isConnected():return
            pos=frame['position']; prev=frames[max(0,i-1)]['position']
            frames_count=max(1,round((1.0 if frame['action']==WAIT else move_seconds)*60))
            for k in range(frames_count):
                t=(k+1)/frames_count
                x=prev[0]*(1-t)+pos[0]*t+.5; y=prev[1]*(1-t)+pos[1]*t+.5
                p.resetBasePositionAndOrientation(robot,(x,y,.27),(0,0,0,1))
                p.resetBasePositionAndOrientation(pkg,(x,y,.65 if frame['carrying'] else -1),(0,0,0,1))
                p.stepSimulation(); time.sleep(1/60)
            text_id=p.addUserDebugText(f"{frame['stage']} | Wait {frame['wait_seconds']}/5 sec | Carrying {frame['carrying']}",
                [1,11,.7],textColorRGB=[.06,.14,.5],textSize=1.1,lifeTime=0,replaceItemUniqueId=text_id)
        print('Mission replay finished; Ctrl+C to close PyBullet')
        while p.isConnected():p.stepSimulation();time.sleep(1/60)
    except KeyboardInterrupt:pass
    finally:
        if p.isConnected():p.disconnect()
