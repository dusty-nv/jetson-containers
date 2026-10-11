#!/usr/bin/env bash
# Install TensorRT from either .deb or .tar.gz
set -eux

echo "Installing TensorRT ${TENSORRT_VERSION}..."

# If no URL provided, skip
if [ -z "${TENSORRT_URL:-}" ]; then
    echo "TENSORRT_URL not provided, skipping TensorRT installation"
    exit 0
fi

cd ${TMP:-/tmp}

# Determine installation method based on URL extension
if [ "${TENSORRT_URL##*.}" = "deb" ]; then
    echo "Installing TensorRT from .deb package..."

    # Download the .deb file
    wget ${WGET_FLAGS} "${TENSORRT_URL}"

    # Install the repository package
    dpkg -i *.deb

    # Copy keyring if it exists
    if [ -d /var/nv-tensorrt-local-repo-* ]; then
        cp /var/nv-tensorrt-local-repo-*/nv-tensorrt-*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || true
    fi

    # Update and install TensorRT packages
    apt-get update

    # Install specified packages or defaults
    PACKAGES="${TENSORRT_PACKAGES:-tensorrt tensorrt-libs python3-libnvinfer-dev}"
    apt-get install -y --no-install-recommends ${PACKAGES}

    # Cleanup
    if [ -n "${TENSORRT_DEB:-}" ]; then
        dpkg -P "${TENSORRT_DEB}" 2>/dev/null || true
    fi
    rm -rf /var/lib/apt/lists/*
    apt-get clean
    rm -rf /tmp/*.deb
    rm -rf /*.deb

elif [ "${TENSORRT_URL%.tar.gz}" != "$TENSORRT_URL" ] || [ "${TENSORRT_URL%.tgz}" != "$TENSORRT_URL" ]; then
    echo "Installing TensorRT from .tar.gz archive..."

    # Download the tar.gz file
    FILENAME=$(basename "${TENSORRT_URL}")
    wget ${WGET_FLAGS} -O "${FILENAME}" "${TENSORRT_URL}"

    # Extract
    mkdir -p tensorrt-extracted
    tar -xzf "${FILENAME}" -C tensorrt-extracted --strip-components=1
    cd tensorrt-extracted

    # Detect architecture for library paths
    if [ "${CUDA_ARCH}" = "tegra-aarch64" ]; then
        # Jetson: use targets/aarch64-linux
        LIB_DIR="/usr/local/cuda/targets/aarch64-linux/lib"
    elif [ "${CUDA_ARCH}" = "aarch64" ] || [ "${IS_SBSA}" = "1" ]; then
        # ARM64 SBSA: use /usr/lib/aarch64-linux-gnu
        LIB_DIR="/usr/lib/aarch64-linux-gnu"
    else
        # x86_64: use /usr/lib/x86_64-linux-gnu
        LIB_DIR="/usr/lib/x86_64-linux-gnu"
    fi

    # Install libraries
    echo "Installing libraries to ${LIB_DIR}..."
    mkdir -p "${LIB_DIR}"
    cp -r lib/*.so* "${LIB_DIR}/" 2>/dev/null || true

    # Install headers
    echo "Installing headers to /usr/include..."
    mkdir -p /usr/include
    cp -r include/* /usr/include/ 2>/dev/null || true

    # Install the TensorRT Python wheel into the active (uv) venv for the image's Python.
    # Ported from NVIDIA-AI-IOT/jetson-containers dev (62b306f4) install_tensorrt.sh.
    # On Jetson (L4T) a missing wheel fails the build; on SBSA/x86 it only warns, as before.
    echo "Installing TensorRT Python bindings..."
    # wheel tag, e.g. cp310, or cp314t for a free-threaded build
    PY_TAG=$(python3 -c 'import sys, sysconfig; print(f"cp{sys.version_info.major}{sys.version_info.minor}" + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""))')
    shopt -s nullglob
    TRT_WHEELS=(python/tensorrt-*-${PY_TAG}-*.whl)
    shopt -u nullglob
    if [ ${#TRT_WHEELS[@]} -gt 0 ]; then
        uv pip install "${TRT_WHEELS[@]}"
    else
        echo "no python/tensorrt-*-${PY_TAG}-*.whl in the TensorRT tarball (${FILENAME}); wheels present:" >&2
        ls python/*.whl >&2 2>/dev/null || echo "  (none)" >&2
        if [ "${IS_TEGRA:-0}" = "1" ]; then
            echo "ERROR: TensorRT Python bindings not installed" >&2
            exit 1
        fi
        echo "WARNING: TensorRT Python bindings not installed (import tensorrt will fail)" >&2
    fi

    # Install bin tools if they exist
    if [ -d "bin" ]; then
        echo "Installing binaries to /usr/bin..."
        cp bin/* /usr/bin/ 2>/dev/null || true
        chmod +x /usr/bin/trt* 2>/dev/null || true
    fi

    # Update library cache
    echo "${LIB_DIR}" > /etc/ld.so.conf.d/tensorrt.conf
    ldconfig

    # Cleanup
    cd ..
    rm -rf tensorrt-extracted "${FILENAME}"

else
    echo "Unknown TensorRT URL format: ${TENSORRT_URL}"
    echo "Expected .deb or .tar.gz file"
    exit 1
fi

# Verify installation
echo "Verifying TensorRT installation..."
if ldconfig -p | grep -q libnvinfer; then
    LIBNVINFER=$(ldconfig -p | grep libnvinfer.so | head -1 | awk '{print $NF}')
    echo "✓ TensorRT installed: ${LIBNVINFER}"

    # Try to get version from library
    if [ -f "${LIBNVINFER}" ]; then
        strings "${LIBNVINFER}" | grep -E "^[0-9]+\.[0-9]+\.[0-9]+" | head -1 || echo "Version: ${TENSORRT_VERSION}"
    fi
else
    echo "⚠ Warning: TensorRT libraries not found in ldconfig"
    echo "Searching for libnvinfer manually..."
    find /usr -name "libnvinfer.so*" 2>/dev/null || true
fi
rm -rf /tmp/*.deb
echo "TensorRT ${TENSORRT_VERSION} installation complete"

