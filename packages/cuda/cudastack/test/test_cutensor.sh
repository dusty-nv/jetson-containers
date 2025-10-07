#!/usr/bin/env bash
set -euo pipefail

# =========================
# cuTENSOR sanity + kernel test (mirrors your cuSPARSELt harness)
# Requires:
#   - nvcc in PATH
#   - TEST_DIR pointing to directory containing test_cutensor.cu (see below)
# Optional:
#   - cuobjdump in PATH (usually in /usr/local/cuda/bin)
# =========================

# --- Helper: extract cuTENSOR "version" from header (compile-time macro) ---
extract_cutensor_header_version() {
  echo "=== cuTENSOR Version (Header) ==="
  local HDR_PATH="$1"
  if [ -f "$HDR_PATH" ]; then
    local VER_LINE
    VER_LINE=$(grep -E '#define[[:space:]]+CUTENSOR_VERSION' "$HDR_PATH" || true)
    if [ -n "${VER_LINE}" ]; then
      # Usually CUTENSOR_VERSION is an integer like 20100 etc.
      local VER_NUM
      VER_NUM=$(echo "$VER_LINE" | awk '{print $3}')
      echo "• CUTENSOR_VERSION macro: ${VER_NUM}"
    else
      echo "• CUTENSOR_VERSION macro not found in: $HDR_PATH"
    fi
  else
    echo "• Header not found: $HDR_PATH"
  fi
  echo ""
}

# --- 1) Find header(s) ---
HDR="$(find /usr/include /usr/local/cuda/include /usr/local/cuda/targets -type f -name cutensor.h 2>/dev/null | head -n1 || true)"
INC_FLAGS=()
[ -n "${HDR}" ] && INC_FLAGS+=(-I"$(dirname "${HDR}")")
[ -d /usr/local/cuda/include ] && INC_FLAGS+=(-I/usr/local/cuda/include)
for d in /usr/local/cuda/targets/*-linux/include; do [ -d "$d" ] && INC_FLAGS+=(-I"$d"); done

echo "=== cuTENSOR Header Detection ==="
if [ -n "${HDR}" ]; then
  echo "• Using header: $HDR"
else
  echo "❌ cutensor.h not found in common include paths"
fi
echo ""

# Header version (if available)
[ -n "${HDR}" ] && extract_cutensor_header_version "$HDR"

# --- 2) Find runtime .so ---
LIB="$(ldconfig -p 2>/dev/null | awk '/libcutensor\.so/{print $NF; exit}')"
[ -z "${LIB:-}" ] && LIB="$(find /usr /usr/local -type f -name 'libcutensor.so*' 2>/dev/null | head -n1 || true)"
if [ -z "${LIB:-}" ]; then
  echo "❌ libcutensor.so not found. Is cuTENSOR installed?"
  exit 1
fi
LIB="$(readlink -f "$LIB")"
LIBDIR="$(dirname "$LIB")"

echo "=== cuTENSOR Runtime Library ==="
echo "• Library: $LIB"
if dpkg -S "$LIB" >/dev/null 2>&1; then
  PKG="$(dpkg -S "$LIB" | cut -d: -f1 | head -n1)"
  VER="$(dpkg -l "$PKG" 2>/dev/null | awk '/^ii/{print $3}')"
  [ -n "$PKG" ] && echo "• Package: $PKG ${VER:+($VER)}"
fi
echo ""

# --- 3) Compose link + rpath flags ---
LIB_FLAGS=(-L"$LIBDIR")
for d in /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/local/cuda/targets/*-linux/lib; do
  [ -d "$d" ] && LIB_FLAGS+=(-L"$d")
done

RPATH_FLAGS=(-Xlinker -rpath="$LIBDIR")
for d in /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/local/cuda/targets/*-linux/lib; do
  [ -d "$d" ] && RPATH_FLAGS+=(-Xlinker -rpath="$d")
done

# --- 4) Ensure cuobjdump in PATH for arch inspection ---
command -v cuobjdump >/dev/null 2>&1 || export PATH="/usr/local/cuda/bin:$PATH"

# --- 5) Inspect supported SM / PTX archs in libcutensor ---
echo "=== cuTENSOR Architecture Scan (cuobjdump) ==="
if ! command -v cuobjdump >/dev/null 2>&1; then
  echo "⚠️  cuobjdump not found; skipping SASS/PTX scan"
else
  echo "• Using: $LIB"
  ARCHS_SASS="$(cuobjdump --list-elf "$LIB" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u || true)"
  ARCHS_PTX="$(cuobjdump --dump-ptx "$LIB" 2>/dev/null | grep -oE 'compute_[0-9]+' | sort -u || true)"
  [ -n "$ARCHS_SASS" ] && echo "• SASS: $ARCHS_SASS"
  [ -n "$ARCHS_PTX" ]  && echo "• PTX : $ARCHS_PTX"
  CNT=$(printf "%s\n" $ARCHS_SASS | grep -c 'sm_' || true)
  echo "📊 Summary: Library has ${CNT} SASS arch(es)"
  if echo "$ARCHS_SASS" | grep -q 'sm_110' || echo "$ARCHS_PTX" | grep -q 'compute_110'; then
    echo "✅ Thor (sm_110) support: YES"
  else
    echo "❌ Thor (sm_110) support: NO"
  fi
fi
echo ""

# --- 6) Build the test ---
: "${TEST_DIR:?Please export TEST_DIR to the directory containing test_cutensor.cu}"

echo "=== Building cuTENSOR test ==="
OUT="/tmp/test_cutensor"
nvcc "${TEST_DIR}/test_cutensor.cu" \
    "${INC_FLAGS[@]}" "${LIB_FLAGS[@]}" \
    -lcutensor -lcudart \
    "${RPATH_FLAGS[@]}" \
    -O2 -std=c++17 \
    -Wno-deprecated-gpu-targets \
    -Wno-deprecated-declarations \
    -o "${OUT}"

echo "• Built: ${OUT}"
echo ""

# --- 7) Run the test binary ---
echo "🚀 === Running cuTENSOR test ==="
"${OUT}"
