#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${VTK_VERSION} --depth=1 --recursive https://gitlab.kitware.com/vtk/vtk /opt/vtk  || \
git clone --depth=1 --recursive https://gitlab.kitware.com/vtk/vtk   /opt/vtk

# Navigate to the vtk repository directory
cd /opt/vtk
mkdir /opt/vtk/build
cd /opt/vtk/build
export MAX_JOBS=$(nproc)
export PYBIN=/usr/bin/python3
cmake -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DVTK_BUILD_TESTING=OFF \
      -DVTK_BUILD_DOCUMENTATION=OFF \
      -DVTK_BUILD_EXAMPLES=OFF \
      -DVTK_DATA_EXCLUDE_FROM_ALL:BOOL=ON \
      -DVTK_MODULE_ENABLE_VTK_PythonInterpreter:STRING=NO \
      -DVTK_MODULE_ENABLE_VTK_WebCore:STRING=YES \
      -DVTK_MODULE_ENABLE_VTK_WebGLExporter:STRING=YES \
      -DVTK_MODULE_ENABLE_VTK_WebPython:STRING=YES \
      -DVTK_WHEEL_BUILD=ON \
      -DVTK_PYTHON_VERSION=3 \
      -DVTK_WRAP_PYTHON=ON \
      -DVTK_OPENGL_HAS_EGL=False \
      -DPython3_EXECUTABLE=$PYBIN ../
ninja

uv pip install wheel
# Build the wheel
python3 setup.py bdist_wheel --dist-dir=$PIP_WHEEL_DIR --verbose
uv pip install $PIP_WHEEL_DIR/vtk*.whl
cd /opt/vtk
# Optionally, upload the wheel using Twine (if configured)
twine upload --verbose $PIP_WHEEL_DIR/vtk*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
