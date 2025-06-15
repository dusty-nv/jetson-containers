#!/usr/bin/env bash
echo "Installing LLVM $LLVM_VERSION"
set -ex

# This would remove apt dependencies, and llvm's version is usually newer
#apt purge -y --autoremove libllvm* libclang* clang-* clangd-* lldb-* lld-* libomp-*
   
# Force overwrites to prevent package conflicts
#echo 'Dpkg::Options {"--force-overwrite";};' > /etc/apt/apt.conf.d/99_force_overwrite

# Download LLVM installer
wget $WGET_FLAGS https://apt.llvm.org/llvm.sh
chmod +x llvm.sh

# Install LLVM and keep cmake
bash llvm.sh $LLVM_VERSION all
bash /tmp/cmake/install.sh

apt-get clean
rm -rf /var/lib/apt/lists/*

# Redirect symlinks to the right version
update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-$LLVM_VERSION 100
update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-$LLVM_VERSION 100
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-$LLVM_VERSION 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-$LLVM_VERSION 100
update-alternatives --install /usr/bin/opt opt /usr/bin/opt-$LLVM_VERSION 100
update-alternatives --install /usr/bin/llc llc /usr/bin/llc-$LLVM_VERSION 100

