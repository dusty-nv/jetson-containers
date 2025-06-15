#!/usr/bin/env bash
set -ex

ls /usr/bin/clang*
ls /usr/bin/llvm*

#ls -R /usr/lib/llvm*
printf "\n"

clang --version
clang++ --version
clang-format --version

llvm-config --prefix
llvm-config --targets-built
llvm-config --version

env | grep LLVM