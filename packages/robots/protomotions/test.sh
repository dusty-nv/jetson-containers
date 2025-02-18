#!/bin/bash
python3 - <<EOF
print('testing ProtoMotions...')

import genesis as gs
import torch
from protomotions.simulator.genesis.config import GenesisSimulatorConfig, GenesisSimParams
from protomotions.simulator.genesis.simulator import GenesisSimulator


print('ProtoMotions OK\\n')
EOF
