#!/bin/bash
python3 - <<EOF
print('testing Isaac-GR00T...')
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# get the data config
data_config = DATA_CONFIG_MAP["gr1_arms_only"]

# get the modality configs and transforms
modality_config = data_config.modality_config()
transforms = data_config.transform()

# This is a LeRobotSingleDataset object that loads the data from the given dataset path.
dataset = LeRobotSingleDataset(
    dataset_path="demo_data/robot_sim.PickNPlace",
    modality_configs=modality_config,
    transforms=transforms,
    embodiment_tag=EmbodimentTag.GR1, # the embodiment to use
)

# This is an example of how to access the data.
dataset[5]

print('Isaac-GR00T OK\\n')
EOF
