#!/usr/bin/env bash
set -exu

echo "Detected architecture: ${CUDA_ARCH}"
echo "IS_SBSA: ${IS_SBSA}"

if [[ "$CUDA_ARCH" == "aarch64" ]] || [[ "$IS_SBSA" == "True" ]]; then
  #  https://developer.download.nvidia.com/compute/cusparselt/0.8.0/local_installers/cusparselt-local-repo-ubuntu2404-0.8.0_0.8.0-1_arm64.deb

  # 1) Install the local repo .deb
  DEB=cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}_${CUSPARSELT_VERSION}-1_arm64.deb
  wget -q "https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/${DEB}"
  dpkg -i "${DEB}"

  # 2) Install the GPG key that the package asked for
  cp /var/cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}/cusparselt-local-*-keyring.gpg /usr/share/keyrings/

  # 3) Update APT
  apt-get update

  # 4) Pick CUDA major automatically (expects nvcc in PATH)
  CUDA_MAJOR=$(nvcc --version | awk -F'release ' '/release/{print $2}' | cut -d. -f1)
  # Fallback if nvcc not available:
  : "${CUDA_MAJOR:=13}"

  # 5) Prefer the CUDA-major-suffixed package names if present
  if apt-cache show "libcusparselt0-cuda-${CUDA_MAJOR}" >/dev/null 2>&1; then
    PKG_RT="libcusparselt0-cuda-${CUDA_MAJOR}"
    PKG_DEV="libcusparselt0-dev-cuda-${CUDA_MAJOR}"
  else
    # Fallback: read package names from the repo's Packages file
    REPO_DIR="$(dirname "$(dpkg -L cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION} | grep '/Packages$')")"
    PKG_RT=$(awk -F': ' '/^Package:/{p=$2} /^Version:/ {v=$2} /^$/ {if(p ~ /^libcusparselt0/){print p}; p=""}' "${REPO_DIR}/Packages" | head -n1)
    PKG_DEV=$(awk -F': ' '/^Package:/{p=$2} /^Version:/ {v=$2} /^$/ {if(p ~ /^libcusparselt0-dev/){print p}; p=""}' "${REPO_DIR}/Packages" | head -n1)
  fi

  echo "Installing: ${PKG_RT} ${PKG_DEV}"
  apt-get install -y "${PKG_RT}" "${PKG_DEV}"

  # 6) Verify
  # 6.1) What got installed?
  dpkg -l | grep -Ei 'cusparselt|libcusparselt' || true
  for p in $(dpkg -l | awk '/libcusparselt/{print $2}'); do
    echo "== $p =="; dpkg -L "$p" | grep -E 'libcusparseLt\.so' || true
  done

  # 6.2) Find every copy on disk
  find /usr /usr/local -type f -name 'libcusparseLt.so*' 2>/dev/null | sort -u | xargs -r ls -l

  # 6.3) If it’s not in the linker cache yet, add CUDA lib dirs & refresh
  echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf
  echo "/usr/local/cuda/targets/aarch64-linux/lib" >> /etc/ld.so.conf.d/cuda.conf
  ldconfig

  # 6.4) Pick the real file and resolve to its canonical path
  LIB=$(ldconfig -p | awk '/libcusparseLt\.so/{print $NF; exit}')
  # Fallback if ldconfig doesn’t list it yet:
  [ -z "$LIB" ] && LIB=$(find /usr /usr/local -type f -name 'libcusparseLt.so.*' 2>/dev/null | head -n1)
  readlink -f "$LIB"

  # 6.5) Now inspect supported architectures (SASS/PTX) on the real file
  command -v cuobjdump >/dev/null 2>&1 || export PATH=/usr/local/cuda/bin:$PATH
  cuobjdump --list-elf "$(readlink -f "$LIB")" | grep -E 'sm_|compute_' | sort -u

elif [[ "$CUDA_ARCH" == "tegra-aarch64" ]]; then
  # Install from .tar.xz for Jetson Orin @ 22.04 and 24.04
  wget $WGET_FLAGS "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}.4_cuda12-archive.tar.xz"
  #                 https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.8.0.4_cuda12-archive.tar.xz
  tar -xJf libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}.0-archive.tar.xz
  cd libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}.0-archive

  if ls *.deb 1>/dev/null 2>&1; then
    dpkg -i *.deb || apt-get -f install -y
  else
    cp -v lib/* /usr/lib/aarch64-linux-gnu/ || cp -v lib/* /usr/local/cuda/lib64/
    cp -v include/* /usr/local/cuda/include/

    ldconfig -p | grep libcusparseLt || true
  fi

else
  wget $WGET_FLAGS \
    "https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}_1.0-1_amd64.deb"
  dpkg -i cusparselt-local-*.deb
  cp /var/cusparselt-local-*/cusparselt-*-keyring.gpg /usr/share/keyrings/
  apt-get update
  apt-get -y install libcusparselt0 libcusparselt-dev
fi

# Clean up (only apt cache; won't touch manual lib copies)
rm -rf /var/lib/apt/lists/*
apt-get clean
