#!/usr/bin/env bash
set -euo pipefail

# Function to extract cuSPARSELt version from library
extract_cusparselt_version() {
    local lib_path="$1"
    echo "=== cuSPARSELt Version Detection ==="

    # Read version from the file we saved during build
    if [ -f "/tmp/CUSPARSELT_VER" ]; then
        local saved_version=$(cat /tmp/CUSPARSELT_VER)
        echo "• Build version: $saved_version"
    else
        echo "• Build version: Not found"
    fi

    # Show library file info
    if [ -f "$lib_path" ]; then
        echo "• Library file: $(basename "$lib_path")"
        echo "• File size: $(ls -lh "$lib_path" | awk '{print $5}')"
        echo "• Last modified: $(ls -l "$lib_path" | awk '{print $6, $7, $8}')"
    fi

    echo "=================================="
    echo ""
}

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
    -o /tmp/test_cusparselt

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

  # Extract cuSPARSELt specific version
  extract_cusparselt_version "$LIB"

  # If dpkg manages it, show package+version (won't error if not)
  if dpkg -S "$LIB" >/dev/null 2>&1; then
    PKG="$(dpkg -S "$LIB" | cut -d: -f1 | head -n1)"
    VER="$(dpkg -l "$PKG" 2>/dev/null | awk '/^ii/{print $3}')"
    [ -n "$PKG" ] && echo "• Package: $PKG ${VER:+($VER)}"
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
/tmp/test_cusparselt
