"""Guaranteed-complete A* mission demo, to test station logic before retraining PPO."""
import argparse
from warehouse.mission import baseline_actions,rollout,WAIT,STATIONS
from warehouse.mission_view import play2d,play3d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('check','2d','3d'),default='check')
    args=ap.parse_args()
    actions=baseline_actions(); frames,reward=rollout(actions)
    assert frames[-1]['success'],'Mission did not finish'
    waits=sum(a==WAIT for a in actions)
    assert waits==10, f'Expected 10 wait actions, got {waits}'
    print('Stages: pickup',STATIONS[0],'-> drop-off',STATIONS[1],'-> goal',STATIONS[2])
    print(f'Completed={frames[-1]["success"]} moves={len(actions)-waits} waits={waits} steps={len(actions)} reward={reward:.1f}')
    if args.mode=='2d':play2d(frames)
    elif args.mode=='3d':play3d(frames)
if __name__=='__main__':main()
