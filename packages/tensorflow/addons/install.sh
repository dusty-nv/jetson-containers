#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of tensorflow-graphics ${TENSORFLOW_GRAPHICS}"
	exit 1
fi

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 17
ln -sf /usr/bin/llvm-config-* /usr/bin/llvm-config
ln -s /usr/bin/clang-1* /usr/bin/clang

pip3 install --no-cache-dir --verbose tensorflow-graphics-gpu==${TENSORFLOW_GRAPHICS_VERSION}