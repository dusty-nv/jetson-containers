#!/usr/bin/env bash
python3 -c 'import robomimic;  print("robomimic version: ", robomimic.__version__)'
python3 -c 'from robomimic.envs.env_robosuite import EnvRobosuite'
