#!/usr/bin/env bash
# downloads, builds, and installs ROS 2 packages from source in $ROS_WORKSPACE
# usage examples:
#   ros2_install.sh xacro teleop_twist_joy
#   ros2_install.sh https://github.com/ros-perception/image_pipeline.git
set -euo pipefail

# ---- Base env (if present) ----------------------------------------------------
[ -f /ros_environment.sh ] && source /ros_environment.sh || true

export MAKEFLAGS="-j $(nproc)"
export ROS_PACKAGE_PATH="${AMENT_PREFIX_PATH:-}"
export PYTHONNOUSERSITE=1
export COLCON_TRACE="${COLCON_TRACE:-0}"

# ---- Workspace & flags --------------------------------------------------------
ROS_WORKSPACE="${ROS_WORKSPACE:=${ROS_ROOT:-/opt/ros/unknown}}"
ROSDEP_SKIP_KEYS="${ROSDEP_SKIP_KEYS:-} gazebo11 libgazebo11-dev libopencv-dev libopencv-contrib-dev libopencv-imgproc-dev python-opencv python3-opencv"
ROS_INSTALL_FLAGS="--deps --exclude RPP --rosdistro ${ROS_DISTRO:-jazzy} ${ROS_INSTALL_FLAGS:-}"
COLCON_FLAGS="--base-paths src --event-handlers console_direct+ ${COLCON_FLAGS:-}"

if [ "${ROS_WORKSPACE}" = "${ROS_ROOT:-}" ]; then
  COLCON_FLAGS="--merge-install ${COLCON_FLAGS}"
else
  COLCON_FLAGS="--symlink-install ${COLCON_FLAGS}"
fi

# ---- Optional virtualenv activation ------------------------------------------
VENV_PATH="${VENV_PATH:-/opt/venv}"
if [ -d "$VENV_PATH" ]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
  # allow seeing system dist-packages if needed
  PYVER="$(python - <<'PY'
import sys; print(".".join(map(str, sys.version_info[:2])))
PY
)"
  export PYTHONPATH="$VENV_PATH/lib/python${PYVER}/site-packages:/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "ROS 2 installing into: ${ROS_WORKSPACE}"
echo "Args: ${*:-<none>}"
echo "Python: $(command -v python)  ($(python -V 2>&1 || true))"
echo "Pip:    $(command -v pip || true)"
echo "Colcon: $(command -v colcon || true)"
mkdir -p "${ROS_WORKSPACE}/src"
set -x

# ---- Fetch sources ------------------------------------------------------------
cd "${ROS_WORKSPACE}"

if [ $# -ge 1 ] && [[ "$1" == http* ]]; then
  SOURCE="git_clone"
  ROS_BRANCH_OPT=""
  [ -n "${ROS_BRANCH:-}" ] && ROS_BRANCH_OPT="--branch ${ROS_BRANCH}"

  cd "${ROS_WORKSPACE}/src"
  git clone --recursive --depth=1 ${ROS_BRANCH_OPT} "$1"

  # INSTALL_PREFIX marker replacement for isaac_ros repos
  if [[ "$1" == *isaac_ros* ]]; then
    cd "$(basename "$1" .git)"
    find . -type f -name "CMakeLists.txt" -print0 | \
      xargs -0 sed -i'' -e 's|<INSTALL_PREFIX>|{CMAKE_INSTALL_PREFIX}|g'
  fi

  cd "${ROS_WORKSPACE}"
  rosinstall_list="$(basename "$1").rosinstall"
  rosinstall_generator ${ROS_INSTALL_FLAGS} --from-path src > "${rosinstall_list}" || \
  rosinstall_generator ${ROS_INSTALL_FLAGS} --from-path src --upstream > "${rosinstall_list}" || true
else
  SOURCE="rosinstall_generator"
  rosinstall_list="ros2.${ROS_DISTRO:-jazzy}.rosinstall"
  if [ $# -ge 1 ]; then
    rosinstall_generator ${ROS_INSTALL_FLAGS} "$@" > "${rosinstall_list}"
  else
    # no args -> nothing to fetch; still allow building already-present src/
    : > "${rosinstall_list}"
  fi
fi

if [ -s "${rosinstall_list}" ]; then
  cat "${rosinstall_list}"
  vcs import --skip-existing src/ < "${rosinstall_list}"
fi

# ---- Ensure dynamic python3 shebangs in generated scripts ---------------------
# Adds [build]\nexecutable=/usr/bin/env python3 if missing.
find "${ROS_WORKSPACE}/src" -name setup.cfg -type f -print0 | while IFS= read -r -d '' f; do
  grep -q '^\[build\]' "$f" || {
    printf '\n[build]\nexecutable=/usr/bin/env python3\n' >> "$f"
    echo ">> added [build] block to $f"
  }
done

# ---- Resolve system deps via rosdep (apt) ------------------------------------
apt-get update
rosdep init || true
rosdep update --rosdistro "${ROS_DISTRO:-jazzy}"
rosdep install -y \
  --ignore-src \
  --from-paths src \
  --rosdistro "${ROS_DISTRO:-jazzy}" \
  --skip-keys "${ROSDEP_SKIP_KEYS}"

rm -rf /var/lib/apt/lists/*
apt-get clean

# ---- Build --------------------------------------------------------------------
colcon build ${COLCON_FLAGS} --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

set +x

# ---- Persist environment on container start -----------------------------------
# Add venv activation and workspace overlay to /ros_environment.sh if available.
if [ -f /ros_environment.sh ] && grep -q 'function ros_source_env' /ros_environment.sh; then
  if [ -d "$VENV_PATH" ] && ! grep -q "$VENV_PATH/bin/activate" /ros_environment.sh; then
    echo "Adding venv activation to /ros_environment.sh"
    tac /ros_environment.sh | sed -e "5isource $VENV_PATH/bin/activate" | tac > /tmp/ros_environment.sh \
      && mv /tmp/ros_environment.sh /ros_environment.sh
  fi
  if ! grep -q "$ROS_WORKSPACE/install/setup.bash" /ros_environment.sh; then
    echo "Adding $ROS_WORKSPACE/install/setup.bash to /ros_environment.sh"
    tac /ros_environment.sh | sed -e "5iros_source_env $ROS_WORKSPACE/install/setup.bash" | tac > /tmp/ros_environment.sh \
      && mv /tmp/ros_environment.sh /ros_environment.sh
  fi
  chmod +x /ros_environment.sh
else
  echo "WARNING: /ros_environment.sh is missing or invalid. Skipping insertion."
fi

echo ""
[ -f /ros_environment.sh ] && cat /ros_environment.sh || true

# ---- Post-check for requested packages (only when using rosinstall_generator) -
if [ "${SOURCE}" = "rosinstall_generator" ] && [ $# -ge 1 ]; then
  # shellcheck disable=SC1091
  [ -d "$VENV_PATH" ] && source "$VENV_PATH/bin/activate" || true
  # shellcheck disable=SC1091
  source "${ROS_WORKSPACE}/install/setup.bash"

  ros_packages="$(ros2 pkg list || true)"
  echo "ROS 2 packages installed:"
  echo "$ros_packages"

  # verify all requested package names appear in the list
  missing=0
  for p in "$@"; do
    if ! echo "$ros_packages" | grep -qx "$p"; then
      echo "NOT FOUND: $p"
      missing=1
    fi
  done
  [ $missing -eq 0 ] || exit 1
fi
