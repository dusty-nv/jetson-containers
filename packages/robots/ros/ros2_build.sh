#!/usr/bin/env bash
# This script builds a ROS2 distribution from source, or installs
# a cached build from jetson-ai-lab (unless FORCE_BUILD=on)
#
# ROS_DISTRO, ROS_ROOT, ROS_PACKAGE environment variables are set
# by the dockerfile or config.py (ex. ROS_ROOT=/opt/ros/humble)
set -euo pipefail

FORCE_BUILD="${FORCE_BUILD:=off}"
BUILD_CACHE="ros-$ROS_DISTRO-$ROS_PACKAGE"
export CUDA_HOME="/usr/local/cuda"
export NVCC_PATH="$CUDA_HOME/bin/nvcc"
SEPARATOR="********************************************************"

print_log() {
  printf "\n$SEPARATOR\n$1\n$SEPARATOR\n\n"
}

print_log " ROS2 $ROS_DISTRO installer ($(uname -m))

   ROS_DISTRO=$ROS_DISTRO
   ROS_PACKAGE=$ROS_PACKAGE
   ROS_ROOT=$ROS_ROOT
   FORCE_BUILD=$FORCE_BUILD
   BUILD_CACHE=$TAR_INDEX_URL/$BUILD_CACHE.tar.gz
   VENV_PATH=$VENV_PATH"

# ------------------------------------------------------------------------------
# Apt repos & base tools
# ------------------------------------------------------------------------------
apt-get update
apt-get install -y --no-install-recommends \
  curl wget gnupg2 lsb-release ca-certificates

curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
| tee /etc/apt/sources.list.d/ros2.list > /dev/null
apt-get update

maybe_install () {
  local pkg="$1"
  local cand
  cand="$(apt-cache policy "$pkg" | awk '/Candidate:/ {print $2}')"
  if [ "${cand:-"(none)"}" != "(none)" ]; then
    apt-get install -y --no-install-recommends "$pkg"
  else
    echo ">> Skipping $pkg: not available"
  fi
}

# ------------------------------------------------------------------------------
# System development libs/headers (keep on apt)
# ------------------------------------------------------------------------------
apt-get install -y --no-install-recommends \
  build-essential cmake pkg-config \
  libeigen3-dev libbullet-dev libpython3-dev \
  libasio-dev libtinyxml2-dev libcunit1-dev \
  libacl1-dev libssl-dev libxaw7-dev libfreetype-dev

maybe_install rti-connext-dds-*
maybe_install rti-connext-dds-*-ros
# TODO: other RMWs as needed

# ------------------------------------------------------------------------------
# Create & activate virtualenv (use system interpreter for ROS ABI compatibility)
# ------------------------------------------------------------------------------
# ensure colcon doesn't try to build the venv
touch "$VENV_PATH/COLCON_IGNORE" || true
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# Prefer venv’s Python packages; include system dist-packages if needed
# (keeps behavior similar to the original script’s PYTHONPATH inclusion)
export PYTHONPATH="$VENV_PATH/lib/python${PYTHON_VERSION}/site-packages:/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"

which python; python -V
which pip; uv run pip --version
which colcon || echo "colcon not found"
echo $VENV_PATH

# ------------------------------------------------------------------------------
# Python tooling into the venv (pip, colcon, ros tooling, test deps)
# ------------------------------------------------------------------------------

print_log "Installing Python tools via pip"
uv pip install --upgrade pip setuptools wheel

uv pip install --upgrade \
  colcon-common-extensions \
  vcstool rosinstall_generator rosdep \
  pytest pytest-cov pytest-repeat pytest-rerunfailures \
  flake8 \
  flake8-blind-except flake8-builtins flake8-class-newline \
  flake8-comprehensions flake8-deprecated flake8-docstrings \
  flake8-import-order flake8-quotes \
  argcomplete \
  lark scikit-build

# ------------------------------------------------------------------------------
# Ensure ROS IDL tools see EmPy 3.x (workaround for 'em.BUFFERED_OPT' error)
# ------------------------------------------------------------------------------
# - Some environments install 'em' (EmPy 4+) by default, which breaks rosidl_*.
# - We remove both 'em' and 'empy' and reinstall EmPy 3.3.4 into the venv.
uv pip uninstall em empy || true
uv pip install --no-cache-dir "empy==3.3.4"

# Freeze venv-only requirements (for reproducibility/caching)
mkdir -p "$ROS_ROOT"

# ------------------------------------------------------------------------------
# Restore specific cmake/numpy versions if your environment requires it
# ------------------------------------------------------------------------------
bash /tmp/cmake/install.sh
bash /tmp/numpy/install.sh

# ------------------------------------------------------------------------------
# Workaround: remove other Python versions if they cause CMake NumPy discovery issues
# ------------------------------------------------------------------------------
apt purge -y python3.9 libpython3.9* || echo "python3.9 not found, skipping removal"
ls -ll /usr/bin/python* || true

# ------------------------------------------------------------------------------
# Prepare source tree
# ------------------------------------------------------------------------------
mkdir -p "${ROS_ROOT}/src"
cd "${ROS_ROOT}"

# extra rosdep entries
ROSDEP_DIR="/etc/ros/rosdep/sources.list.d"
mkdir -p "$ROSDEP_DIR" || true
cp "$TMP/rosdeps.yml" "$ROSDEP_DIR/extra-rosdeps.yml"
echo "yaml file://$ROSDEP_DIR/extra-rosdeps.yml" | tee "$ROSDEP_DIR/00-extras.list"

# Humble/Iron on bionic patches
if { [ "$ROS_DISTRO" = "humble" ] || [ "$ROS_DISTRO" = "iron" ]; } && \
   [ "$(lsb_release --codename --short)" = "bionic" ]; then
  # skip unavailable keys
  SKIP_KEYS="$SKIP_KEYS rti-connext-dds-6.0.1 ignition-cmake2 ignition-math6"

  # newer GCC
  apt-get install -y --no-install-recommends gcc-8 g++-8
  export CC="/usr/bin/gcc-8" CXX="/usr/bin/g++-8"
  echo "CC=$CC CXX=$CXX"

  # yaml-cpp 0.6.0
  git -C /tmp clone -b yaml-cpp-0.6.0 https://github.com/jbeder/yaml-cpp.git
  cmake -S /tmp/yaml-cpp -B /tmp/yaml-cpp/BUILD -DBUILD_SHARED_LIBS=ON
  cmake --build /tmp/yaml-cpp/BUILD --parallel "$(nproc --ignore=1)"
  cmake --install /tmp/yaml-cpp/BUILD
  rm -rf /tmp/yaml-cpp
fi

# ------------------------------------------------------------------------------
# Attempt cached install
# ------------------------------------------------------------------------------
rosdep_install() {
  # Install apt deps recorded in rosdeps.txt
  xargs -a "$ROS_ROOT/rosdeps.txt" apt-get install -y --no-install-suggests --no-install-recommends
}

cached_install() {
  TARPACK_PREFIX=$ROS_ROOT tarpack install "$BUILD_CACHE" || return $?
  rosdep_install || return $?
  CACHED_INSTALL="yes"
}

if [ "$FORCE_BUILD" != "on" ]; then
  cached_install || true
  if [ "${CACHED_INSTALL:-}" = "yes" ]; then
    print_log "INSTALLED ROS2 $ROS_DISTRO from cache:\n  $TAR_INDEX_URL/$BUILD_CACHE.tar.gz"
    exit 0
  fi
fi

# ------------------------------------------------------------------------------
# Build from source
# ------------------------------------------------------------------------------
print_log " BUILDING ROS2 $ROS_DISTRO from source ($ROS_PACKAGE)"
set -x

# Generate sources list
rosinstall_generator --deps --rosdistro "${ROS_DISTRO}" "${ROS_PACKAGE}" \
  launch_xml \
  launch_yaml \
  launch_testing \
  launch_testing_ament_cmake \
  demo_nodes_cpp \
  demo_nodes_py \
  example_interfaces \
  camera_calibration_parsers \
  camera_info_manager \
  cv_bridge \
  v4l2_camera \
  vision_opencv \
  vision_msgs \
  image_geometry \
  image_pipeline \
  image_transport \
  compressed_image_transport \
  compressed_depth_image_transport \
  rosbag2_storage_mcap \
  rmw_fastrtps \
> "ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall"

cat "ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall"
vcs import src < "ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall"

# Replace ament_cmake with branch-matched version
rm -rf "${ROS_ROOT}/src/ament_cmake"
git -C "${ROS_ROOT}/src" clone https://github.com/ament/ament_cmake -b "${ROS_DISTRO}"

# Ensure dynamically-resolved python3 shebangs for generated scripts
find src -name setup.cfg -type f -print0 | while IFS= read -r -d '' f; do
  grep -q '^\[build\]' "$f" || {
    printf '\n[build]\nexecutable=/usr/bin/env python3\n' >> "$f"
    echo ">> added [build] block to $f"
  }
done

# rosdep (first-time init is machine-wide; OK in containers)
rosdep init || true
rosdep update

# Resolve and record apt deps (filtering some conflicting libs)
rosdep keys \
  --from-paths src \
  --ignore-src src \
  --rosdistro "$ROS_DISTRO" \
| xargs rosdep resolve \
| grep -v \# \
| grep -v opencv \
| grep -v pybind11 \
> rosdeps.txt

rosdep_install

# Restore cmake/numpy pin (if needed by your env)
bash /tmp/cmake/install.sh
bash /tmp/numpy/install.sh

# Build with colcon from the venv
colcon build \
  --merge-install \
  --cmake-args \
  -Wno-dev -Wno-deprecated \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_DEFAULT_CMP0148=OLD
  # -DCMAKE_WARN_DEPRECATED=OFF

# ------------------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------------------
rm -rf "${ROS_ROOT}/src" "${ROS_ROOT}/log" "${ROS_ROOT}/build"
rm -f "${ROS_ROOT}"/*.rosinstall

# apt cleanup
rm -rf /var/lib/apt/lists/*
apt-get clean

# Try uploading cached build
tarpack upload "$BUILD_CACHE" "$ROS_ROOT" || echo "failed to upload tarball"

which python; python -V
which uv pip; uv run pip --version
which colcon || echo "colcon not found"
echo $VENV_PATH
