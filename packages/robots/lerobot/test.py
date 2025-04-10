#!/usr/bin/env python3
import lerobot

print(lerobot.available_envs)
print(lerobot.available_tasks_per_env)
print(lerobot.available_datasets)
print(lerobot.available_datasets_per_env)
print(lerobot.available_real_world_datasets)
print(lerobot.available_policies)
print(lerobot.available_policies_per_env)
print(lerobot.available_robots)
        
print('lerobot version:', lerobot.__version__)