#!/usr/bin/env python3
import time
import imageio
import mimicgen
import numpy as np
import robosuite as rs

print('MIMICGEN    ', mimicgen.__version__)
print('ROBOSUITE   ', rs.__version__)
print('ENVIRONMENTS', list(rs.ALL_ENVIRONMENTS))
print('CONTROLLERS ', list(rs.ALL_CONTROLLERS))
print('ROBOTS      ', list(rs.ALL_ROBOTS))
print('GRIPPERS    ', list(rs.ALL_GRIPPERS), '\n')

# create environment instance
env = rs.make(
    env_name="Stack_D0", 
    robots="Panda",
    gripper_types="PandaGripper",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="frontview",
    camera_widths=512,
    camera_heights=512,
)

# reset the environment
env.reset()
frames = []

for i in range(10):
    time_begin = time.perf_counter()
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    render_time = time.perf_counter() - time_begin
    ##env.render()  # render on display
    image = obs.get('frontview_image')
    #image = np.flip(image, axis=0)
    #frames.append(image)
    print(f"frame {i}  ({render_time*1000:.2f} ms, {1.0/render_time:.1f} fps)")
    print(f"  observations:", list(obs.keys()))
    print(f"  image:", image.shape, image.dtype)
    print(f"  reward:  {reward}")
    print(f"  done:    {done}")
    print(f"  state:   {info}")

#imageio.mimsave("robosuite.mp4", np.stack(frames), fps=15)
print("\nIgnore:  OpenGL.raw.EGL._errors.EGLError: <exception str() failed>")
