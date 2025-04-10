#!/usr/bin/env python3
import os
import time
import json
import pprint
import imageio
import argparse
import mimicgen
import multiprocessing

import numpy as np
import robosuite as rs


print('MIMICGEN    ', mimicgen.__version__)
print('ROBOSUITE   ', rs.__version__)
print('ENVIRONMENTS', list(rs.ALL_ENVIRONMENTS))
print('CONTROLLERS ', list(rs.ALL_CONTROLLERS))
print('ROBOTS      ', list(rs.ALL_ROBOTS))
print('GRIPPERS    ', list(rs.ALL_GRIPPERS), '\n')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
parser.add_argument('--tasks', type=str, nargs='+', default=['Stack_D0', 'Stack_D1'], choices=list(rs.ALL_ENVIRONMENTS), help="one or more tasks to generate episodes for")
parser.add_argument('--robots', type=str, nargs='+', default=['Panda'], choices=list(rs.ALL_ROBOTS), help="one or more robots to generate episodes for")
parser.add_argument('--grippers', type=str, nargs='+', default=['PandaGripper'], choices=list(rs.ALL_GRIPPERS), help="one or more grippers to generate episodes for")

parser.add_argument('--cameras', type=str, nargs='+', default=['agentview', 'frontview', 'robot0_eye_in_hand'], help="one or more camera views to render")
parser.add_argument('--camera-width', type=int, default=512, help="the width (in pixels) of the camera")
parser.add_argument('--camera-height', type=int, default=512, help="the height (in pixels) of the camera")

parser.add_argument('--frames', type=int, default=60, help="the number of frames to generate per configuration")
parser.add_argument('--parallel', type=int, default=4, help="the number of parallel processes to run")
parser.add_argument('--output', type=str, default="/data/sim/mimicgen/test", help="output directory of the demo videos")

args = parser.parse_args()
print(args)


def generate(task='Stack_D0', robot='Panda', gripper='PandaGripper', 
             camera='frontview', camera_width=512, camera_height=512, 
             frames=20, output=None, **kwargs):
    """
    Generate a sample video of the environment
    """
    env = rs.make(
        env_name=task, 
        robots=robot,
        gripper_types=gripper,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera,
        camera_widths=camera_width,
        camera_heights=camera_height,
    )
    
    env.reset()
    images = []

    time_begin = time.perf_counter()
    
    for i in range(frames):
        action = np.random.randn(env.robots[0].dof) # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment
        ##env.render()  # render on display
        #image = env.sim.render(mode='offscreen', width=camera_width, height=camera_height, camera_name=camera)
        image = obs.get(f'{camera}_image') 
        if any(x in camera for x in ['agentview', 'frontview', 'sideview']):
            image = np.flip(image, axis=0)
        images.append(image)
     
    render_time = (time.perf_counter() - time_begin) / frames
    
    stats = dict(
        task = task,
        robot = robot,
        gripper = gripper,
        camera = camera,
        render_fps = 1/render_time,
        render_time = render_time,
        observations = list(obs.keys()),
        image = images[-1].shape,
        reward = reward,
        done = done,
        info = info
    )
    
    stats_name = f"{task}-{robot}-{gripper}-{camera}" 
    
    if output:
        stats['output'] = os.path.join(output, stats_name)
    
    print(f"\nProcess {os.getpid()} - done generating scene:\n\n{pprint.pformat(stats, indent=2, sort_dicts=False)}")
    
    if not output:
        return

    stats_path = os.path.join(output, stats_name + '.json')
    image_path = os.path.join(output, stats_name + '.jpg')
    video_path = os.path.join(output, stats_name + '.mp4')

    os.makedirs(output, exist_ok=True)
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
        
    imageio.imwrite(image_path, images[0])
    imageio.mimsave(video_path, np.stack(images), fps=20)
    

for key in ['tasks', 'robots', 'grippers', 'cameras']:
    arg = getattr(args, key)
    if isinstance(arg, str):
        setattr(args, key, [arg])

if args.parallel:
    pool = multiprocessing.Pool(args.parallel)
    pool_results = []
    
generations = 0

for task in args.tasks:
    for robot in args.robots:
        for gripper in args.grippers:
            for camera in args.cameras:
                config = dict(task=task, robot=robot, gripper=gripper, camera=camera, **vars(args))
                if args.parallel:
                    pool_results.append(pool.apply_async(
                        generate, kwds=config
                    ))
                else:
                    generate(**config)
                generations += 1

if args.parallel:
    for result in pool_results:
        result.wait()
                      
print(f"\nMIMICGEN SCENE GENERATIONS COMPLETED ({generations}/{generations})\n")
print(f"   * tasks={args.tasks}  robots={args.robots}  grippers={args.grippers}  cameras={args.cameras}")
print(f"   * images/video saved to:  {args.output}")
print(f"   * ignore EGL errors on exit below\n")


