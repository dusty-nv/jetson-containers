#!/usr/bin/env bash

set -exu

echo "Detected architecture: ${CUDA_ARCH}"
echo "IS_SBSA: ${IS_SBSA:-False}"

CUDA_MAJOR=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p')
: "${CUDA_MAJOR:=${CUDA_VERSION_MAJOR:-13}}"

# Returns 0 (true) if $1 >= $2 (semver-aware comparison)
version_gte() {
  [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

if version_gte "${CUSPARSELT_VERSION}" "0.9.0"; then
  # ── 0.9.0+ : unified APT install via CUDA repo keyring ─────────────────────
  if [ "$CUDA_ARCH" = "tegra-aarch64" ] && [ "${CUDA_INSTALLED_VERSION:-0}" -lt 132 ]; then
    REPO_ARCH="arm64"
  elif [ "$(uname -m)" = "aarch64" ]; then
    REPO_ARCH="sbsa"
  else
    REPO_ARCH="x86_64"
  fi

  cd /tmp
  wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb" \
    -O cuda-keyring.deb
  dpkg -i cuda-keyring.deb
  apt-get update

  PKG_RT="libcusparselt0-cuda-${CUDA_MAJOR}"
  PKG_DEV="libcusparselt0-dev-cuda-${CUDA_MAJOR}"
  if ! apt-cache show "${PKG_RT}" >/dev/null 2>&1; then
    PKG_RT="libcusparselt0"
    PKG_DEV="libcusparselt0-dev"
  fi

  RT_APT_VER=$(apt-cache madison "${PKG_RT}" 2>/dev/null \
    | awk -v ver="${CUSPARSELT_VERSION}" '$3 ~ ver {gsub(/^ +| +$/, "", $3); print $3; exit}')
  DEV_APT_VER=$(apt-cache madison "${PKG_DEV}" 2>/dev/null \
    | awk -v ver="${CUSPARSELT_VERSION}" '$3 ~ ver {gsub(/^ +| +$/, "", $3); print $3; exit}')

  if [ -n "${RT_APT_VER}" ] && [ -n "${DEV_APT_VER}" ]; then
    echo "Pinning cuSPARSELt to: ${PKG_RT}=${RT_APT_VER} ${PKG_DEV}=${DEV_APT_VER}"
    apt-get install -y --no-install-recommends "${PKG_RT}=${RT_APT_VER}" "${PKG_DEV}=${DEV_APT_VER}"
  else
    echo "Exact version ${CUSPARSELT_VERSION} not found in repo, installing latest"
    apt-get install -y --no-install-recommends "${PKG_RT}" "${PKG_DEV}"
  fi

  ldconfig

  # Remove cuda-keyring to prevent the global NVIDIA repo from propagating
  dpkg --purge cuda-keyring 2>/dev/null || true
  rm -f /etc/apt/sources.list.d/cuda-*-keyring.list
  rm -f /etc/apt/preferences.d/cuda-repository-pin-600
  rm -f /usr/share/keyrings/cuda-archive-keyring.gpg

else
  # ── 0.8.x and earlier ───────────────────────────────────────────────────────
  if [ "$CUDA_ARCH" = "aarch64" ] || [ "${IS_SBSA:-False}" = "True" ]; then
    #  https://developer.download.nvidia.com/compute/cusparselt/0.8.0/local_installers/cusparselt-local-repo-ubuntu2404-0.8.0_0.8.0-1_arm64.deb
    #  https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-repo-ubuntu2404-0.7.1_1.0-1_arm64.deb (*Thor not supported)
    #  https://developer.download.nvidia.com/compute/cusparselt/0.7.0/local_installers/cusparselt-local-repo-ubuntu2404-0.7.0_1.0-1_arm64.deb (*Thor not supported)

    # 1) Install the local repo .deb
    DEB=cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}_${CUSPARSELT_VERSION}-1_arm64.deb
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate \
      -O "${DEB}" "https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/${DEB}"
    dpkg -i "${DEB}"

    # 2) Install the GPG key that the package asked for
    cp /var/cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}/cusparselt-local-*-keyring.gpg /usr/share/keyrings/

    # 3) Update APT
    apt-get update

    # 4) Prefer the CUDA-major-suffixed package names if present
    if apt-cache show "libcusparselt0-cuda-${CUDA_MAJOR}" >/dev/null 2>&1; then
      PKG_RT="libcusparselt0-cuda-${CUDA_MAJOR}"
      PKG_DEV="libcusparselt0-dev-cuda-${CUDA_MAJOR}"
    else
      # Fallback: read package names from the repo's Packages file
      REPO_DIR="$(dirname "$(dpkg -L cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION} | grep '/Packages$')")"
      PKG_RT=$(awk -F': ' '/^Package:/{p=$2} /^$/ {if(p ~ /^libcusparselt0/){print p}; p=""}' "${REPO_DIR}/Packages" | head -n1)
      PKG_DEV=$(awk -F': ' '/^Package:/{p=$2} /^$/ {if(p ~ /^libcusparselt0-dev/){print p}; p=""}' "${REPO_DIR}/Packages" | head -n1)
    fi

    echo "Installing: ${PKG_RT} ${PKG_DEV}"
    apt-get install -y "${PKG_RT}" "${PKG_DEV}"

    # 5) Verify
    dpkg -l | grep -Ei 'cusparselt|libcusparselt' || true
    for p in $(dpkg -l | awk '/libcusparselt/{print $2}'); do
      echo "== $p =="; dpkg -L "$p" | grep -E 'libcusparseLt\.so' || true
    done

    find /usr /usr/local -type f -name 'libcusparseLt.so*' 2>/dev/null | sort -u | xargs -r ls -l

    echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf
    echo "/usr/local/cuda/targets/aarch64-linux/lib" >> /etc/ld.so.conf.d/cuda.conf
    ldconfig

    LIB=$(ldconfig -p | awk '/libcusparseLt\.so/{print $NF; exit}')
    [ -z "$LIB" ] && LIB=$(find /usr /usr/local -type f -name 'libcusparseLt.so.*' 2>/dev/null | head -n1)
    readlink -f "$LIB"

    command -v cuobjdump >/dev/null 2>&1 || export PATH=/usr/local/cuda/bin:$PATH
    cuobjdump --list-elf "$(readlink -f "$LIB")" | grep -E 'sm_|compute_' | sort -u

  elif [ "$CUDA_ARCH" = "tegra-aarch64" ]; then
    # Install from .tar.xz for Jetson Orin @ 22.04 and 24.04 (for CUDA 12)
    #  https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.8.0.4_cuda12-archive.tar.xz
    #  https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
    #  https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.7.0.0-archive.tar.xz

    # Fallback CUDA major to 12 for JP6.x
    : "${CUDA_MAJOR:=12}"

    # Determine the patch suffix based on CUSPARSELT_VERSION
    # Pattern: 0.8.1 -> .1, 0.8.0 -> .4, 0.7.1 -> .0, 0.7.0 -> .0
    if [ "${CUSPARSELT_VERSION}" = "0.8.1" ]; then
      PATCH_SUFFIX="1"
    elif [ "${CUSPARSELT_VERSION}" = "0.8.0" ]; then
      PATCH_SUFFIX="4"
    elif [ "${CUSPARSELT_VERSION}" = "0.7.1" ]; then
      PATCH_SUFFIX="0"
    elif [ "${CUSPARSELT_VERSION}" = "0.7.0" ]; then
      PATCH_SUFFIX="0"
    else
      PATCH_SUFFIX="0"
    fi

    VER="${CUSPARSELT_VERSION}.${PATCH_SUFFIX}"
    echo "CUSPARSELT_VERSION: ${CUSPARSELT_VERSION}"
    echo "PATCH_SUFFIX: ${PATCH_SUFFIX}"
    echo "Final VER: ${VER}"
    BASE_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64"

    # Different versions have different archive naming patterns
    if [ "${CUSPARSELT_VERSION}" = "0.8.1" ] || [ "${CUSPARSELT_VERSION}" = "0.8.0" ]; then
      ARCHIVE="libcusparse_lt-linux-aarch64-${VER}_cuda${CUDA_MAJOR}-archive.tar.xz"
    else
      ARCHIVE="libcusparse_lt-linux-aarch64-${VER}-archive.tar.xz"
    fi

    WORK="${TMP:-/tmp}/cusparselt"
    EXTRACT="${WORK}/extract"
    mkdir -p "$WORK" "$EXTRACT"
    cd "$WORK"

    echo "Downloading $ARCHIVE ..."
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate \
      -O "$ARCHIVE" "${BASE_URL}/${ARCHIVE}"

    echo "Extracting $ARCHIVE ..."
    tar -xJf "$ARCHIVE" --strip-components=1 -C "$EXTRACT"
    cd "$EXTRACT"

    # Install headers under a CUDA-majored include prefix (JP6: /usr/include/libcusparseLt/12)
    HDR_DST="/usr/include/libcusparseLt/${CUDA_MAJOR}"
    install -d "$HDR_DST"
    install -m 0644 include/cusparseLt*.h "$HDR_DST/"

    # Install libs into CUDA Tegra target lib dir
    LIB_DST="/usr/local/cuda/targets/aarch64-linux/lib"
    install -d "$LIB_DST"
    install -m 0644 lib/libcusparseLt.so* "$LIB_DST/"

    echo "$LIB_DST" >/etc/ld.so.conf.d/cusparselt.conf
    ldconfig

    echo "Installed headers to $HDR_DST"
    echo "Installed libraries to $LIB_DST"

    command -v cuobjdump >/dev/null 2>&1 || export PATH=/usr/local/cuda/bin:$PATH
    if [ -x "$(command -v cuobjdump)" ]; then
      echo "🔍 cuSPARSELt SASS/PTX targets:"
      cuobjdump --list-elf "$LIB_DST"/libcusparseLt.so* 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u || true
      cuobjdump --dump-ptx "$LIB_DST"/libcusparseLt.so* 2>/dev/null | grep -oE 'compute_[0-9]+' | sort -u || true
    fi

  else
    # x86_64
    # wget $WGET_FLAGS \
    #  "https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}_1.0-1_amd64.deb"
    wget $WGET_FLAGS \
      "https://developer.download.nvidia.com/compute/cusparselt/0.8.0/local_installers/cusparselt-local-repo-${DISTRO}-0.8.0_1.0-1_amd64.deb"

    dpkg -i cusparselt-local-*.deb
    cp /var/cusparselt-local-*/cusparselt-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install libcusparselt0 libcusparselt-dev
  fi
fi

# ── Common cleanup ─────────────────────────────────────────────────────────────
rm -f /tmp/cuda-keyring.deb
rm -rf /var/lib/apt/lists/*
apt-get clean
rm -rf /tmp/*.deb
rm -rf /*.deb

# Save version info for runtime testing
echo "${CUSPARSELT_VERSION}" > /tmp/CUSPARSELT_VER
echo "cuSPARSELt ${CUSPARSELT_VERSION} installed successfully"
