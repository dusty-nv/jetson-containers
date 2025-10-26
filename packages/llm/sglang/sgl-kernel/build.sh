#!/usr/bin/env bash
set -x

# ===== Pretty logging helpers =====
is_tty() { [[ -t 1 ]]; }
if is_tty && command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RESET="$(tput sgr0)"
  CYAN="$(tput setaf 6)"; GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"; MAGENTA="$(tput setaf 5)"
else
  BOLD=""; DIM=""; RESET=""; CYAN=""; GREEN=""; YELLOW=""; MAGENTA=""
fi

hr()      { printf "${DIM}%*s${RESET}\n" "$(tput cols 2>/dev/null || echo 80)" "" | tr " " "─"; }
section() { printf "\n${BOLD}${CYAN}🔹 %s${RESET}\n" "$1"; }
kv()      { printf "  ${DIM}%s${RESET}: %s\n" "$1" "$2"; }
ok()      { printf "${GREEN}✔${RESET} %s\n" "$1"; }
warn()    { printf "${YELLOW}⚠${RESET} %s\n" "$1"; }

# ===== Required vars =====
: "${SGL_KERNEL_VERSION:?SGL_KERNEL_VERSION must be set}"
: "${PIP_WHEEL_DIR:?PIP_WHEEL_DIR must be set}"

# ===== Deps =====
apt update && apt install -y ccache
uv pip install -U compressed-tensors decord2 ninja setuptools wheel numpy scikit-build-core twine

REPO_URL="https://github.com/sgl-project/sglang"
REPO_DIR="/opt/sglang"

section "Build: SGLang ${SGL_KERNEL_VERSION}"
echo "Tagged branch not found; cloning default branch"

rm -rf "${REPO_DIR}"
# first try: clone specific branch/tag
git clone --recursive --depth 1 --branch ${SGL_KERNEL_BRANCH} ${REPO_URL} ${REPO_DIR} || \
{ # fallback: try to init/fetch the specific commit
  mkdir ${REPO_DIR} && cd ${REPO_DIR} && \
  git init && git remote add origin ${REPO_URL} && \
  git fetch --depth 1 origin ${SGL_KERNEL_BRANCH} && git checkout FETCH_HEAD
} || \
{ # last resort: clone default branch
  cd "$(dirname ${REPO_DIR})" && rm -rf "$(basename ${REPO_DIR})" && \
  git clone --recursive --depth 1 ${REPO_URL} ${REPO_DIR}
}

cd "${REPO_DIR}/sgl-kernel" || exit 1
sed -i 's/==/>=/g' pyproject.toml

section "Configuring parallelism"
if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
    export MAX_JOBS=6
else
    export MAX_JOBS=32
    export CMAKE_BUILD_PARALLEL_LEVEL=32
    export CPLUS_INCLUDE_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl
fi
export NVCC_THREADS=2
export CUDA_NVCC_FLAGS="-Xcudafe --threads=2"

printf "\n${BOLD}${MAGENTA}🚀 Build Plan${RESET}\n"
kv "MAX_JOBS" "${MAX_JOBS}"
kv "CMAKE_BUILD_PARALLEL_LEVEL" "${CMAKE_BUILD_PARALLEL_LEVEL:-unset}"
kv "NINJAFLAGS" "${NINJAFLAGS:-unset}"

# You already compute/export these above in your pipeline; we just print them nicely.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export CUDA_VERSION="${CUDA_VERSION}"

kv "TORCH_CUDA_ARCH_LIST" "${TORCH_CUDA_ARCH_LIST:-unset}"
kv "CUDA_VERSION" "${CUDA_VERSION:-unset}"
hr

section "Toolchain versions"
printf "${DIM}"
echo "nvcc:";     nvcc --version || true
echo "gcc:";      gcc  --version | sed -n '1,2p' || true
echo "g++:";      g++  --version | sed -n '1,2p' || true
echo "cmake:";    cmake --version | sed -n '1,2p' || true
echo "python3:";  python3 --version || true
echo "ninja:";    ninja --version || true
printf "${RESET}"
hr

section "Building wheel → ${PIP_WHEEL_DIR}"
ok "Generating sgl-kernel wheel (no build isolation, Ninja)…"

cd "${REPO_DIR}/sgl-kernel" || exit 1

if [[ "$(uname -m)" == "aarch64" && "${TORCH_CUDA_ARCH_LIST}" = "8.7" ]]; then
  # Override parallel build env vars for SM 87 on Jetson devices
  # Note: make sure you have sufficient swap space set to avoid out of memory problems
  MAX_JOBS=$(nproc)
  export MAX_JOBS=$MAX_JOBS
  export CMAKE_BUILD_PARALLEL_LEVEL=$((MAX_JOBS/2))
  # Apply patch for SM 87
  if ! git apply -p1 ${TMP}/sm_87-${SGL_KERNEL_VERSION}.diff ;then
    warn "Patch for SM 8.7 FAILED!" && exit 1
  else
    ok "Patch for SM 8.7 applied successfully!"
  fi
fi

cat CMakeLists.txt

if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
    uv build --wheel --no-build-isolation . --out-dir "${PIP_WHEEL_DIR}" \
    --config-settings=cmake.args="-G;Ninja" \
    --config-settings=cmake.define.TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    --config-settings=cmake.define.CUDA_VERSION="${CUDA_VERSION}" \
    --config-settings=cmake.define.SGL_KERNEL_ENABLE_FA3=1 \
    --config-settings=cmake.define.ENABLE_BELOW_SM90=ON \
    --config-settings=cmake.define.CMAKE_POLICY_VERSION_MINIMUM=3.5
else
    uv build --wheel --no-build-isolation . --out-dir "${PIP_WHEEL_DIR}" \
      --config-settings=cmake.args="-G;Ninja" \
      --config-settings=cmake.define.TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
      --config-settings=cmake.define.CUDA_VERSION="${CUDA_VERSION}" \
      --config-settings=cmake.define.SGL_KERNEL_ENABLE_BF16=1 \
      --config-settings=cmake.define.SGL_KERNEL_ENABLE_FP8=1 \
      --config-settings=cmake.define.SGL_KERNEL_ENABLE_FP4=1 \
      --config-settings=cmake.define.SGL_KERNEL_ENABLE_FA3=1 \
      --config-settings=cmake.define.SGL_KERNEL_ENABLE_SM90A=1 \
      --config-settings=cmake.define.SGL_KERNEL_ENABLE_SM100A=1 \
      --config-settings=cmake.define.ENABLE_BELOW_SM90=ON \
      --config-settings=cmake.define.CMAKE_POLICY_VERSION_MINIMUM=3.5
fi

section "Installing wheel"
uv pip install "${PIP_WHEEL_DIR}/sgl"*.whl && ok "Wheel installed"

cd "${REPO_DIR}" || exit 1

section "Uploading wheel"
twine upload --verbose "${PIP_WHEEL_DIR}/sgl"*.whl \
  && ok "Uploaded to ${TWINE_REPOSITORY_URL:-<default repo>}" \
  || warn "Failed to upload wheel to ${TWINE_REPOSITORY_URL:-<unset>}"
