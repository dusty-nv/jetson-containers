#!/usr/bin/env bash
echo "Installing LLVM $LLVM_VERSION"
set -ex

# This would remove apt dependencies, and llvm's version is usually newer
#apt purge -y --autoremove libllvm* libclang* clang-* clangd-* lldb-* lld-* libomp-*
   
# Force overwrites to prevent package conflicts
#echo 'Dpkg::Options {"--force-overwrite";};' > /etc/apt/apt.conf.d/99_force_overwrite

# Download LLVM installer and run it
wget $WGET_FLAGS https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
bash llvm.sh $LLVM_VERSION all

# Redirect symlinks to the right version
update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-$LLVM_VERSION 100
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-$LLVM_VERSION 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-$LLVM_VERSION 100
update-alternatives --install /usr/bin/opt opt /usr/bin/opt-$LLVM_VERSION 100
update-alternatives --install /usr/bin/llc llc /usr/bin/llc-$LLVM_VERSION 100