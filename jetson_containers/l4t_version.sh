
#!/bin/bash
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

get_l4t_version() {
    # 1. If argument is provided, use it
    if [ -n "$1" ]; then
        echo "$1"
        return
    fi

    # 2. If environment variable is set, use it
    if [ -n "$L4T_VERSION" ]; then
        echo "$L4T_VERSION"
        return
    fi

    # 3. If not on Jetson, return default
    ARCH=$(uname -m)
    if [ "$ARCH" != "aarch64" ] || ! uname -a | grep -qi "tegra"; then
        echo "36.4.3"
        return
    fi

    # 4. If version file does not exist, return default
    VERSION_FILE="/etc/nv_tegra_release"
    if [ ! -f "$VERSION_FILE" ]; then
        echo "36.4.3"
        return
    fi

    # 5. Parse version from file
    L4T_VERSION_STRING=$(head -n 1 "$VERSION_FILE")
    L4T_RELEASE=$(echo "$L4T_VERSION_STRING" | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
    L4T_REVISION=$(echo "$L4T_VERSION_STRING" | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')
	L4T_REVISION_MAJOR=${L4T_REVISION:0:1}
	L4T_REVISION_MINOR=${L4T_REVISION:2:1}

    echo "$L4T_RELEASE.$L4T_REVISION"
}

L4T_VERSION=$(get_l4t_version)
echo "| L4T_VERSION=$L4T_VERSION"

TEGRA="tegra"
if [ -z "${SYSTEM_ARCH}" ]; then
  ARCH=$(uname -m)

  if [ "$ARCH" = "aarch64" ]; then
    echo "| ### ARM64 architecture detected"
    if uname -a | grep -qi "$TEGRA"; then
      SYSTEM_ARCH="$TEGRA-$ARCH"
      echo "| ### Jetson Detected"
    else
      echo "| ### SBSA Detected"
      SYSTEM_ARCH="$ARCH"
    fi
  elif [ $ARCH = "x86_64" ]; then
    echo "| ### x86 Detected"
    SYSTEM_ARCH="$ARCH"
  else
	echo "| ### 🚫 Unsupported architecture:  $ARCH"
	exit 1
  fi
fi
echo "| SYSTEM_ARCH=$SYSTEM_ARCH"

IS_SBSA=$([ -z "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep nvgpu)" ] && echo 1 || echo 0)
echo "| IS_SBSA=$IS_SBSA"

COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')
echo "| COMPUTE_CAPABILITY=$COMPUTE_CAPABILITY"