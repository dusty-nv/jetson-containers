#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
if [ ! -d /opt/nerfstudio ]; then
    echo "Cloning Nerfstudio version ${NERFSTUDIO_VERSION}"
    git clone --branch=v${NERFSTUDIO_VERSION} --depth=1 --recursive https://github.com/nerfstudio-project/nerfstudio /opt/nerfstudio ||
    git clone --depth=1 --recursive https://github.com/nerfstudio-project/nerfstudio /opt/nerfstudio
fi

echo "Installing build dependencies..."
uv pip install scikit-build-core ninja # ninja is often needed too

cd /opt/nerfstudio
uv pip install --reinstall blinker
sed -i 's/==/>=/g' pyproject.toml
uv pip install cmake open3d
uv pip install --reinstall blinker
uv build --wheel . --out-dir /opt/nerfstudio/wheels  # Create the wheel package

# Verify the contents of the /opt directory
ls /opt/nerfstudio/wheels
# Return to the root directory
cd /
uv pip install manifold3d vhacdx openexr
uv pip install --reinstall blinker
uv pip install /opt/nerfstudio/wheels/nerfstudio*.whl

ns-install-cli --mode install

cd /opt/nerfstudio
uv pip install -U --force-reinstall opencv-contrib-python
# Optionally upload to a repository using Twine
twine upload --verbose /opt/nerfstudio/wheels/nerfstudio*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
