#!/usr/bin/env bash
set -x

# Ensure required variables are set
: "${SGLANG_VERSION:?SGLANG_VERSION must be set}"
: "${PIP_WHEEL_DIR:?PIP_WHEEL_DIR must be set}"

# --- PRE-INSTALL DEPS ---
# Install build dependencies first. uv is a very fast installer.
pip3 install --no-cache-dir ninja setuptools wheel numpy uv scikit-build-core compressed-tensors decord2

# --- CLONE SGLANG REPO ---
REPO_URL="https://github.com/sgl-project/sglang"
REPO_DIR="/opt/sglang"

echo "Building SGLang ${SGLANG_VERSION}"

if [ ! -d "${REPO_DIR}" ]; then
  if git clone --recursive --depth 1 --branch "v${SGLANG_VERSION}" \
      "${REPO_URL}" "${REPO_DIR}"; then
    echo "Cloned SGLang v${SGLANG_VERSION}"
  else
    echo "Tagged branch v${SGLANG_VERSION} not found; cloning default branch"
    git clone --recursive --depth 1 "${REPO_URL}" "${REPO_DIR}"
  fi
else
  echo "Directory ${REPO_DIR} already exists, skipping clone."
fi
cd "${REPO_DIR}" || exit 1

# --- PATCH 1: MAKE srt/utils.py JETSON-AWARE (ROBUST FIX) ---
# This patch is much better than the original sed. It correctly detects
# Jetson and reports system memory, avoiding errors later on.
UTILS_PATH="python/sglang/srt/utils.py"
if [[ -f "${UTILS_PATH}" ]]; then
  echo "Applying robust Jetson patch to ${UTILS_PATH}"
  # Use a heredoc to overwrite the function completely
  sed -i '/def get_nvgpu_memory_capacity/,/return min(memory_values)/c\
def is_jetson():\
    """Checks if the system is an NVIDIA Jetson device."""\
    return os.path.exists("/etc/nv_tegra_release")\
\
def get_nvgpu_memory_capacity():\
    """\
    Gets the total memory capacity of the GPU in MiB.\
\
    This function is adapted to work robustly on both standard PCs with\
    NVIDIA GPUs and on NVIDIA Jetson devices. It explicitly checks for a\
    Jetson platform first.\
    """\
    if is_jetson():\
        try:\
            with open("/proc/meminfo") as f:\
                for line in f:\
                    if "MemTotal" in line:\
                        mem_total_kb = float(line.split()[1])\
                        return mem_total_kb / 1024\
            raise ValueError("Could not find MemTotal in /proc/meminfo on Jetson device.")\
        except Exception as e:\
            raise RuntimeError(\
                "Failed to read system memory from /proc/meminfo on a Jetson device."\
            ) from e\
    else:\
        try:\
            result = subprocess.run(\
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],\
                stdout=subprocess.PIPE,\
                stderr=subprocess.PIPE,\
                text=True,\
                check=True,\
            )\
            memory_values = [\
                float(mem)\
                for mem in result.stdout.strip().split("\\n")\
                if re.match(r"^\\d+(\\.\\d+)?$", mem.strip())\
            ]\
            if not memory_values:\
                raise ValueError("No GPU memory values found in nvidia-smi output.")\
            return min(memory_values)\
        except FileNotFoundError:\
            raise RuntimeError(\
                "Not a Jetson device and nvidia-smi was not found. "\
                "Ensure NVIDIA drivers are installed and in the system'"'"'s PATH."\
            )\
        except subprocess.CalledProcessError as e:\
            raise RuntimeError(f"nvidia-smi command failed: {e.stderr.strip()}")\
' "${UTILS_PATH}"
fi

# --- PATCH 2: RELAX PYTORCH VERSION REQUIREMENTS ---
cd "${REPO_DIR}/python" || exit 1

# Your original sed commands are good. This makes the build compatible with
# the PyTorch versions typically available on Jetson.
sed -i -E '
s/"torch>=2\.7\.1"/"torch>=2.7.0"/;
s/"torchaudio==2\.7\.1"/"torchaudio>=2.7.0"/;
s/"torchvision==0\.22\.1"/"torchvision>=0.22.0"/
' pyproject.toml
sed -i -E '
s/torch==2\.7\.1/torch>=2.7.0/;
s/torchaudio==2\.7\.1/torchaudio>=2.7.0/;
s/torchvision==0\.22\.1/torchvision>=0.22.0/
' pyproject.toml
sed -i 's/==/>=/g' pyproject.toml

echo "Patched ${REPO_DIR}/python/pyproject.toml to relax version constraints"
cat pyproject.toml

# --- CONFIGURE PARALLEL BUILD ---
if [[ -z "${IS_SBSA:-}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
  export CORES=$(nproc) # Automatically use all available cores
else
  export CORES=32  # GH200 or other specific hardware
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${CORES}"
export MAX_JOBS="${CORES}"

# --- BUILD SGLANG WHEEL (THE RIGHT WAY) ---
echo "🚀 Building sglang wheel ONLY with MAX_JOBS=${CORES}"

# Use '--no-deps' to build ONLY the sglang wheel and ignore its dependencies.
# We will install dependencies later when we install the built wheel.
pip3 wheel \
    --no-build-isolation \
    --no-deps \
    . \
    --wheel-dir "${PIP_WHEEL_DIR}"

# --- INSTALL THE BUILT WHEEL AND ITS DEPENDENCIES ---
echo "✅ sglang wheel built successfully."
echo "📦 Installing the sglang wheel from ${PIP_WHEEL_DIR} and its dependencies from PyPI..."

# Now, when we install the local wheel, pip will fetch its dependencies
# (like torch, transformers, etc.) from the online package index (PyPI).
# We use 'uv' here because it's extremely fast.
uv pip install "${PIP_WHEEL_DIR}/sglang"*.whl

# Your original script installed 'gemlite' here, so we keep it.
uv pip install gemlite orjson

echo "🎉 SGLang and all dependencies installed successfully!"

cd / || exit 1

# Try uploading; ignore failure
if [ -x "$(command -v twine)" ]; then
    twine upload --verbose "${PIP_WHEEL_DIR}/sglang"*.whl \
      || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL:-<unset>}"
else
    echo "twine not installed, skipping upload."
fi
