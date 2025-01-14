#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
if [ ! -d /opt/nerfstudio ]; then
    echo "Cloning Nerfstudio version ${NERFSTUDIO_VERSION}"
    git clone --branch=v${NERFSTUDIO_VERSION} --depth=1 --recursive https://github.com/nerfstudio-project/nerfstudio /opt/nerfstudio || \
    git clone --depth=1 --recursive https://github.com/nerfstudio-project/nerfstudio /opt/nerfstudio
fi

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/nerfstudio

pip3 wheel . --no-deps --no-build-isolation -w /opt/nerfstudio/wheels  # Create the wheel package

# Verify the contents of the /opt directory
ls /opt/nerfstudio/wheels

# Return to the root directory
cd /
pip3 install tyro appdirs av comet-ml fpsample gsplat imageio ipywidgets jaxtyping jupyterlab mediapy msgpack msgpack-numpy nerfacc nuscenes-devkit open3d opencv-python-headless pathos pyngrok pyquaternion python-socketio pytorch-msssim rawpy scikit-image splines tensorboard tensorly timm torchmetrics trimesh viser wandb xatlas cryptography==38 protobuf==3.20.3

pip3 install --no-cache-dir --verbose /opt/nerfstudio/wheels/nerfstudio*.whl

ns-install-cli --mode install

cd /opt/nerfstudio
# Optionally upload to a repository using Twine
twine upload --verbose /opt/nerfstudio/wheels/nerfstudio*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
