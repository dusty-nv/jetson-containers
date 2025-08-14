#!/usr/bin/env bash
set -euo pipefail

# Function to extract version information from library files
extract_version_info() {
    local lib_path="$1"
    echo "=== cuSPARSELt Version Information ==="

    # Try to get version from package manager
    if command -v dpkg >/dev/null 2>&1; then
        local pkg_info=$(dpkg -S "$lib_path" 2>/dev/null | head -n1 || true)
        if [ -n "$pkg_info" ]; then
            local pkg_name=$(echo "$pkg_info" | cut -d: -f1)
            local pkg_version=$(dpkg -l "$pkg_name" 2>/dev/null | awk '/^ii/{print $3}' || true)
            echo "• Package: $pkg_name ${pkg_version:+($pkg_version)}"
        fi
    fi

    # Try to get version from library file itself
    if [ -f "$lib_path" ]; then
        echo "• Library file: $(basename "$lib_path")"
        echo "• File size: $(ls -lh "$lib_path" | awk '{print $5}')"
        echo "• Last modified: $(ls -l "$lib_path" | awk '{print $6, $7, $8}')"

        # Try to extract version from library symbols
        if command -v nm >/dev/null 2>&1; then
            local version_symbols=$(nm -D "$lib_path" 2>/dev/null | grep -i version || true)
            if [ -n "$version_symbols" ]; then
                echo "• Version symbols found:"
                echo "$version_symbols" | head -5 | sed 's/^/  /'
            fi
        fi
    fi

    # Try to get CUDA version info
    if command -v nvcc >/dev/null 2>&1; then
        local cuda_version=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9][0-9]*\).*/\1/' || true)
        echo "• CUDA version: ${cuda_version:-'Unknown'}"
    fi

    echo "======================================"
    echo ""
}

# Function to detect system architecture and SBSA status
detect_system_info() {
    echo "=== System Architecture Detection ==="

    # Detect CUDA architecture
    local detected_arch="unknown"
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)
        if [[ "$gpu_name" == *"Orin"* ]] || [[ "$gpu_name" == *"Jetson"* ]]; then
            detected_arch="tegra-aarch64"
        elif [[ "$gpu_name" == *"A100"* ]] || [[ "$gpu_name" == *"H100"* ]] || [[ "$gpu_name" == *"L40"* ]]; then
            detected_arch="aarch64"
        else
            detected_arch="x86_64"
        fi
        echo "• GPU: $gpu_name"
    fi
    echo "• Detected CUDA_ARCH: $detected_arch"

    # Detect SBSA status
    local is_sbsa="False"
    if [[ "$detected_arch" == "aarch64" ]] && [[ "$detected_arch" != "tegra-aarch64" ]]; then
        is_sbsa="True"
    fi
    echo "• Detected IS_SBSA: $is_sbsa"

    # Detect distribution
    local detected_distro="unknown"
    if [ -f /etc/os-release ]; then
        detected_distro=$(grep "^ID=" /etc/os-release | cut -d= -f2 | tr -d '"')
    fi
    echo "• Detected DISTRO: $detected_distro"

    echo "====================================="
    echo ""
}

echo "=== cuSPARSELt Test Environment ==="
echo "CUSPARSELT_VERSION: ${CUSPARSELT_VERSION:-'Not set'}"
echo "CUDA_ARCH: ${CUDA_ARCH:-'Not set'}"
echo "IS_SBSA: ${IS_SBSA:-'Not set'}"
echo "DISTRO: ${DISTRO:-'Not set'}"
echo "=================================="
echo ""

# Detect system information
detect_system_info

rm -f /test/test_cusparselt || true

# 1) Find the header (works for /usr/include/libcusparseLt/13/cusparseLt.h, CUDA targets, etc)
HDR=$(find /usr/include /usr/local/cuda -type f -name cusparseLt.h 2>/dev/null | head -n1 || true)
INC_FLAGS=()
[ -n "${HDR}" ] && INC_FLAGS+=(-I"$(dirname "${HDR}")")
[ -d /usr/local/cuda/include ] && INC_FLAGS+=(-I/usr/local/cuda/include)
for d in /usr/local/cuda/targets/*-linux/include; do [ -d "$d" ] && INC_FLAGS+=(-I"$d"); done

# 2) Find the runtime .so and its directory
#    Prefer ldconfig; fallback to searching common roots
LIB=$(ldconfig -p | awk '/libcusparseLt\.so/{print $NF; exit}') || true
[ -z "${LIB:-}" ] && LIB=$(find /usr /usr/local -type f -name 'libcusparseLt.so*' 2>/dev/null | head -n1 || true)
if [ -z "${LIB}" ]; then
  echo "ERROR: libcusparseLt.so not found. Is libcusparselt0-cuda-13 installed?"
  exit 1
fi
LIB=$(readlink -f "$LIB")
LIBDIR=$(dirname "$LIB")

# 3) Compose -L and rpath (keep CUDA dirs too)
LIB_FLAGS=(-L"$LIBDIR")
for d in /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/local/cuda/targets/*-linux/lib; do
  [ -d "$d" ] && LIB_FLAGS+=(-L"$d")
done

RPATH_FLAGS=(-Xlinker -rpath="$LIBDIR")
for d in /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/local/cuda/targets/*-linux/lib; do
  [ -d "$d" ] && RPATH_FLAGS+=(-Xlinker -rpath="$d")
done

# 4) Build and link
nvcc /test/test_cusparselt.cu \
    "${INC_FLAGS[@]}" "${LIB_FLAGS[@]}" \
    -lcusparseLt -lcusparse -lcudart \
    "${RPATH_FLAGS[@]}" \
    -Wno-deprecated-gpu-targets \
    -Wno-deprecated-declarations \
    -o /test/test_cusparselt

# 4) Show supported SM architectures in the library
# 4.0) ensure cuobjdump is available
command -v cuobjdump >/dev/null 2>&1 || export PATH=/usr/local/cuda/bin:$PATH

# 4.1) find the actual lib (prefer ldconfig; fall back to filesystem search)
LIB="$(ldconfig -p 2>/dev/null | awk '/libcusparseLt\.so/{print $NF; exit}')"
if [ -z "${LIB:-}" ]; then
  LIB="$(find /usr /usr/local -type f -name 'libcusparseLt.so*' 2>/dev/null | head -n1 || true)"
fi
if [ -z "${LIB:-}" ]; then
  echo "❌ libcusparseLt not found on this system"
else
  LIB="$(readlink -f "$LIB")"
  echo "• Using: $LIB"
  echo ""

  # Extract detailed version information
  extract_version_info "$LIB"

  # If dpkg manages it, show package+version (won't error if not)
  if dpkg -S "$LIB" >/dev/null 2>&1; then
    PKG="$(dpkg -S "$LIB" | cut -d: -f1 | head -n1)"
    VER="$(dpkg -l "$PKG" 2>/dev/null | awk '/^ii/{print $3}')"
    [ -n "$PKG" ] && echo "• Package: $PKG ${VER:+($VER)}"
  fi

  # Show library file version info
  echo "• Library file: $(basename "$LIB")"
  if [ -f "$LIB" ]; then
    echo "• File size: $(ls -lh "$LIB" | awk '{print $5}')"
    echo "• Last modified: $(ls -l "$LIB" | awk '{print $6, $7, $8}')"
  fi

  if ! command -v cuobjdump >/dev/null 2>&1; then
    echo "⚠️  cuobjdump not found; skipping arch scan"
  else
    echo "🔎 Inspecting supported architectures…"
    # 4.2) SASS targets (compiled code objects)
    ARCHS_SASS="$(cuobjdump --list-elf "$LIB" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u || true)"
    # 4.3) PTX targets (JIT-able targets)
    ARCHS_PTX="$(cuobjdump --dump-ptx "$LIB" 2>/dev/null | grep -oE 'compute_[0-9]+' | sort -u || true)"

    if [ -n "$ARCHS_SASS$ARCHS_PTX" ]; then
      [ -n "$ARCHS_SASS" ] && { echo "• SASS: $ARCHS_SASS"; }
      [ -n "$ARCHS_PTX" ]  && { echo "• PTX : $ARCHS_PTX";  }
      # summary (SASS count is the most meaningful)
      CNT=$(printf "%s\n" $ARCHS_SASS | grep -c 'sm_' || true)
      echo "📊 Summary: Library has ${CNT} SASS arch(es)"

      # Thor support check (accept SASS sm_110 or PTX compute_110)
      if echo "$ARCHS_SASS" | grep -q 'sm_110' || echo "$ARCHS_PTX" | grep -q 'compute_110'; then
        echo "✅ Thor (sm_110) support: YES"
      else
        echo "❌ Thor (sm_110) support: NO"
      fi
    else
      echo "❌ Could not determine supported architectures with cuobjdump"
    fi
  fi
fi

# 5) Run the test
echo ""
echo "🚀 === Running cusparseLt test ==="
/test/test_cusparselt

# 6) Version Summary
echo ""
echo "📋 === Version Summary ==="
echo "Environment:"
echo "  • CUSPARSELT_VERSION: ${CUSPARSELT_VERSION:-'Not set'}"
echo "  • CUDA_ARCH: ${CUDA_ARCH:-'Detected: tegra-aarch64'}"
echo "  • IS_SBSA: ${IS_SBSA:-'Detected: False'}"
echo "  • DISTRO: ${DISTRO:-'Detected: ubuntu'}"
echo "  • GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 'Unknown')"

if [ -n "${LIB:-}" ]; then
  echo "Library:"
  echo "  • Path: $LIB"
  if dpkg -S "$LIB" >/dev/null 2>&1; then
    PKG="$(dpkg -S "$LIB" | cut -d: -f1 | head -n1)"
    VER="$(dpkg -l "$PKG" 2>/dev/null | awk '/^ii/{print $3}')"
    [ -n "$PKG" ] && echo "  • Package: $PKG ${VER:+($VER)}"
  fi
fi

echo "=================================="
