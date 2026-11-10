#!/usr/bin/env python3
"""Real-robot diffusion rollout with rate limits and optional review pauses.

This is a deployment debugging tool, separate from the fixed STEP evaluation plan.
Q/Escape, STOP file, SIGINT/SIGTERM stop motion; each segment ends holding position.
"""
import argparse
import json
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

from diffusion_common import IMAGE, JOINTS, load_policy, seed_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parent/'runs/diffusion_cube/checkpoints/002000')
    p.add_argument('--port',default='/dev/cu.usbmodem5B421342761')
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execution-horizon',type=int,default=8)
    p.add_argument('--continuous',action='store_true',help='Run continuously until max-seconds or operator stop')
    p.add_argument('--release-on-exit',action='store_true',help='Disable all motor torque on exit; support the arm before stopping')
    p.add_argument('--max-relative-target',type=float,default=3.,help='Maximum body target lead over measured position in degrees')
    p.add_argument('--segment-seconds',type=float,default=5)
    p.add_argument('--max-seconds',type=float,default=40)
    p.add_argument('--body-speed',type=float,default=15,help='Maximum commanded degrees per second')
    p.add_argument('--gripper-speed',type=float,default=40,help='Maximum commanded percentage points per second')
    a=p.parse_args()
    if not (0<a.segment_seconds<=10 and 0<a.max_seconds<=180 and 0<a.body_speed<=20 and 0<a.gripper_speed<=60 and 0<a.max_relative_target<=10):
        raise ValueError('Invalid rollout limits')
    a.output.mkdir(parents=True,exist_ok=False)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    (a.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    policy,pre,post=load_policy(a.checkpoint,'mps',a.execution_horizon)
    torch.set_num_threads(8)
    seed_all(42)
    names=[n.removesuffix('.pos') for n in JOINTS]
    bus=SerialMotorsBus(port=a.port,motors={n:Motor(i+1,'sts3215',MotorNormMode.DEGREES if i<5 else MotorNormMode.RANGE_0_100) for i,n in enumerate(names)})
    camera=None;video=None;armed=False;stop=False;frames=0;active=0.;segment=0;records=[];next_snapshot=0.
    result={'success':None,'reason':'setup_error','formal_evaluation':False}
    window='Diffusion robot test - Q to STOP'
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    def hold():
        raw=bus.sync_read('Present_Position',normalize=False)
        bus.sync_write('Goal_Position',raw,normalize=False)
    def capture():
        ok,image=camera.read()
        if not ok:raise RuntimeError('Wrist camera frame lost')
        if image.shape!=(480,640,3):raise RuntimeError('Unexpected camera resolution')
        return image
    def observation(image,state):
        return {'observation.state':torch.tensor([[state[n] for n in names]],dtype=torch.float32),IMAGE:torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).unsqueeze(0).float()/255}
    try:
        bus.connect();bus.calibration=bus.read_calibration()
        (a.output/'calibration_from_registers.json').write_text(json.dumps({n:vars(v) for n,v in bus.calibration.items()},indent=2)+'\n')
        for n in names:
            if bus.read('Operating_Mode',n,normalize=False)!=0:raise RuntimeError('Motor not in position mode: '+n)
        camera=cv2.VideoCapture(a.camera,cv2.CAP_AVFOUNDATION)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640);camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480);camera.set(cv2.CAP_PROP_FPS,30)
        for _ in range(10):image=capture()
        cv2.imwrite(str(a.output/'before.jpg'),image)
        cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,960,720)
        with torch.inference_mode():
            warm=post(policy.select_action(pre(observation(image,bus.sync_read('Present_Position')))))
        if not torch.isfinite(warm).all():raise RuntimeError('Invalid warmup prediction')
        policy.reset();seed_all(42)
        # The gripper was off during connection checks. Apply the same current/torque
        # limits as LeRobot's SO follower before enabling it; body PID is unchanged.
        if not bus.read('Torque_Enable','gripper',normalize=False):
            for key,value in [('Max_Torque_Limit',500),('Protection_Current',250),('Overload_Torque',25)]:
                bus.write(key,'gripper',value,normalize=False)
        hold();armed=True;bus.enable_torque()
        last_goal=bus.sync_read('Present_Position');last_time=time.monotonic();next_health=0.
        video=cv2.VideoWriter(str(a.output/'wrist.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
        if not video.isOpened():raise RuntimeError('Cannot record rollout video')
        with (a.output/'actions.jsonl').open('w') as log,torch.inference_mode():
            while not stop and active<a.max_seconds:
                segment+=1;start=time.monotonic()
                duration = a.max_seconds-active if a.continuous else min(a.segment_seconds,a.max_seconds-active)
                while not stop and time.monotonic()-start<duration:
                    tick=time.monotonic()
                    if (a.output/'STOP').exists():stop=True;break
                    image=capture();state=bus.sync_read('Present_Position',num_retry=2)
                    if active+tick-start>=next_health:
                        temp=bus.sync_read('Present_Temperature',normalize=False)
                        if max(temp.values())>55:raise RuntimeError('Motor temperature above 55 C')
                        next_health=active+tick-start+1
                    prediction=post(policy.select_action(pre(observation(image,state)))).cpu().numpy().reshape(-1)
                    if prediction.shape!=(6,) or not np.isfinite(prediction).all():raise RuntimeError('Invalid model action')
                    now=time.monotonic();dt=min(max(now-last_time,1/30),.1)
                    if stop or now-start>=duration:break
                    goals={}
                    for i,n in enumerate(names):
                        speed=a.body_speed if i<5 else a.gripper_speed
                        goal=float(np.clip(prediction[i],last_goal[n]-speed*dt,last_goal[n]+speed*dt))
                        lag=a.max_relative_target if i<5 else 10.
                        goal=float(np.clip(goal,state[n]-lag,state[n]+lag))
                        if i<5:
                            cal=bus.calibration[n];mid=(cal.range_min+cal.range_max)/2
                            lo=(cal.range_min+12-mid)*360/4095;hi=(cal.range_max-12-mid)*360/4095
                            goal=float(np.clip(goal,lo,hi))
                        else:goal=float(np.clip(goal,0,100))
                        goals[n]=goal
                    bus.sync_write('Goal_Position',goals);last_goal=goals;last_time=now
                    record={'segment':segment,'active_s':active+now-start,'state':state,'predicted':dict(zip(names,prediction.tolist())),'sent':goals,'cycle_s':time.monotonic()-tick}
                    log.write(json.dumps(record)+'\n');log.flush();records.append(record)
                    video.write(image);frames+=1
                    if record['active_s'] >= next_snapshot:
                        latest = a.output/'latest_frame.tmp.jpg'
                        if not cv2.imwrite(str(latest),image):
                            raise RuntimeError('Cannot save supervision frame')
                        latest.replace(a.output/'latest_frame.jpg')
                        status = {'active_s':record['active_s'],'frames':frames,'state':state,'waiting':False}
                        (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
                        next_snapshot=record['active_s']+1
                    cv2.putText(image,f'Test {active+now-start:.1f}s | Q: STOP',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
                    cv2.imshow(window,image)
                    if cv2.waitKey(1)&0xff in (ord('q'),27):stop=True
                    if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:stop=True
                    time.sleep(max(0,1/30-(time.monotonic()-tick)))
                active+=time.monotonic()-start;hold()
                image=capture();cv2.imwrite(str(a.output/f'paused_{segment:02d}.jpg'),image)
                status={'segment':segment,'active_s':active,'frames':frames,'state':bus.sync_read('Present_Position'),'waiting':not stop and not a.continuous and active<a.max_seconds}
                (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
                print('FINISHED' if a.continuous or stop else 'PAUSED',json.dumps(status),flush=True)
                if a.continuous or stop or active>=a.max_seconds:break
                resume=a.output/f'CONTINUE_{segment:02d}'
                wait_started=time.monotonic()
                while not stop and not resume.exists():
                    if (a.output/'STOP').exists() or time.monotonic()-wait_started>120:stop=True;break
                    image=capture();cv2.putText(image,'PAUSED - holding | Q: STOP',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
                    cv2.imshow(window,image)
                    if cv2.waitKey(30)&0xff in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:stop=True
                last_goal=bus.sync_read('Present_Position');last_time=time.monotonic()
        result.update(reason='operator_or_review_stop' if stop else 'timeout',active_s=active,frames=frames,final_state=bus.sync_read('Present_Position'))
    except BaseException as exc:
        result.update(reason='runtime_error',error=repr(exc));raise
    finally:
        try:
            if armed and bus.is_connected:
                if a.release_on_exit:
                    bus.disable_torque(num_retry=3)
                    enabled = bus.sync_read('Torque_Enable', normalize=False, num_retry=3)
                    if any(enabled.values()):
                        raise RuntimeError(f'Torque release could not be confirmed: {enabled}')
                    result['end_state']='All motor torque disabled; arm and gripper released'
                else:
                    hold()
                    result['end_state']='Holding measured position; torque remains enabled'
        finally:
            bus.disconnect(disable_torque=False)
            if camera is not None:camera.release()
            if video is not None:video.release()
            cv2.destroyAllWindows()
            (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print('RESULT',json.dumps(result),flush=True)

if __name__=='__main__':main()
