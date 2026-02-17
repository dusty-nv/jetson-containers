#!/bin/bash
python3 - <<EOF
print('testing Isaac-GR00T...')
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# get the data config
data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]

# get the modality configs and transforms
modality_config = data_config.modality_config()
transforms = data_config.transform()

print('Isaac-GR00T OK\\n')
EOF
