#!/usr/bin/env bash
set -ex

ls /usr/bin/llvm*
#ls -R /usr/lib/llvm*

printf "\n"

llvm-config --prefix
llvm-config --targets-built
llvm-config --version

env | grep LLVM