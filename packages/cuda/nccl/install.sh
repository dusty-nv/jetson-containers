#!/usr/bin/env bash
set -euo pipefail

# --- (preserve any existing header logic) ---
: "${NCCL_VERSION:=2.27.7}"
: "${DISTRO:=ubuntu2204}"
: "${CUDA_MAJOR:=13}"
: "${CUDA_MINOR:=0}"

# Allow a user-supplied download host via MULTIARCH_URL (keeps your WIP compatibility)
# Default to apt.jetson-ai-lab.io (upstream pattern)
: "${MULTIARCH_URL:=https://apt.jetson-ai-lab.io/multiarch}"

echo "Installing NVIDIA NCCL ${NCCL_VERSION}"

# Determine package arch suffix from uname -m and map to expected deb suffix
ARCH="$(uname -m)"
case "$ARCH" in
  aarch64) PKG_ARCH="aarch64" ;;
  arm64)   PKG_ARCH="arm64" ;;
  x86_64)  PKG_ARCH="amd64" ;;
  amd64)   PKG_ARCH="amd64" ;;
  *)       PKG_ARCH="$ARCH" ;;
esac

# If we are on Jetson tegra-aarch64, prefer tarpack build path (existing tarpack/tarpack helper)
if [[ "$PKG_ARCH" == "aarch64" ]] && [[ "$(uname -m)" == "aarch64" ]] && [[ "${CUDA_ARCH:-}" == "tegra-aarch64" || "$(uname -m)" == "aarch64" && -n "${TEGRA:-}" ]]; then
  # Preserve your original tegra handling: if FORCE_BUILD is 'on', error (or force a build path).
  if [[ "${FORCE_BUILD:-}" == "on" ]]; then
    echo "Forcing build of NVIDIA NCCL ${NCCL_VERSION} on Tegra"
    exit 1
  fi

  echo "Detected Tegra (tegra-aarch64) — using tarpack install for nccl-${NCCL_VERSION}"
  tarpack install "nccl-${NCCL_VERSION}"
  exit 0
fi

# Construct filenames to try. Primary name uses explicit CUDA_MAJOR.CUDA_MINOR
DEB_NAME="nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda${CUDA_MAJOR}.${CUDA_MINOR}_1.0-1_${PKG_ARCH}.deb"
DEB_URL="${MULTIARCH_URL}/${DEB_NAME}"
ALT_NAME="nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda${CUDA_MAJOR}.${CUDA_MINOR}_${PKG_ARCH}.deb"
# Also include legacy cuda13.0 naming fallback if needed (keeps some earlier patterns)
LEGACY_NAME_V1="nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda${CUDA_MAJOR}.${CUDA_MINOR}_1.0-1_${PKG_ARCH}.deb"
LEGACY_NAME_V2="nccl-local-repo-${DISTRO}-${NCCL_VERSION}-cuda${CUDA_MAJOR}.${CUDA_MINOR}_${PKG_ARCH}.deb"

echo "Trying to download ${DEB_NAME} from ${MULTIARCH_URL}"
mkdir -p /tmp/nccl
cd /tmp/nccl

# helper to wget with optional auth header (keeps your WIP NCCL_FETCH_AUTH idea)
wget_with_auth() {
  local url="$1"
  if [ -n "${NCCL_FETCH_AUTH:-}" ]; then
    # allow passing "--user user:token" or other wget auth flags via NCCL_FETCH_AUTH
    eval "wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${NCCL_FETCH_AUTH} \"${url}\"" || true
  else
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate "${url}" || true
  fi
}

# 1) Try primary URL
wget_with_auth "${DEB_URL}"

# 2) Fallback to alternate names if primary not downloaded
if [ ! -f "${DEB_NAME}" ]; then
  echo "Primary DEB not found, trying alternate: ${ALT_NAME}"
  wget_with_auth "${MULTIARCH_URL}/${ALT_NAME}"
  if [ -f "${ALT_NAME}" ]; then
    DEB_NAME="${ALT_NAME}"
  fi
fi

# 3) Try some legacy fallback patterns (older scripts used slightly different naming)
if [ ! -f "${DEB_NAME}" ]; then
  echo "Trying legacy name v1: ${LEGACY_NAME_V1}"
  wget_with_auth "${MULTIARCH_URL}/${LEGACY_NAME_V1}"
  if [ -f "${LEGACY_NAME_V1}" ]; then
    DEB_NAME="${LEGACY_NAME_V1}"
  fi
fi

if [ ! -f "${DEB_NAME}" ]; then
  echo "Trying legacy name v2: ${LEGACY_NAME_V2}"
  wget_with_auth "${MULTIARCH_URL}/${LEGACY_NAME_V2}"
  if [ -f "${LEGACY_NAME_V2}" ]; then
    DEB_NAME="${LEGACY_NAME_V2}"
  fi
fi

# If still not found, try any previously-downloaded name in TMP (last resort), else fail
if [ ! -f "${DEB_NAME}" ]; then
  echo "No matching NCCL DEB found in ${MULTIARCH_URL}; listing /tmp/nccl for debugging:"
  ls -la /tmp/nccl || true
  echo "ERROR: NCCL package not found; aborting."
  exit 4
fi

echo "Installing ${DEB_NAME} ..."
dpkg -i "${DEB_NAME}" || true

# Copy any keyring(s) packaged by the nccl-local-repo deb (wildcard to support multiple variants)
# Some local-repo packages place keyrings under /var/nccl-local-repo-<...>/nccl-local-*-keyring.gpg
cp -v /var/nccl-local-repo-*/nccl-local-*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || true

# Update apt and install the library components
apt-get update || true
apt-get -y install libnccl2 libnccl-dev
