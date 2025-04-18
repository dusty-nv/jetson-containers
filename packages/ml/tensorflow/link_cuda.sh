#!/bin/bash
set -ex

prefix=/opt/nvidia/cudnn
arch=$(uname -m)-linux-gnu

cuda_base_path="/usr/local/cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}"
cuda_targets=${cuda_base_path}/targets/$(uname -m)-linux

rm ${cuda_base_path}/include ${cuda_base_path}/lib64
mv ${cuda_targets}/include ${cuda_base_path}/include
mv ${cuda_targets}/lib ${cuda_base_path}/lib
ln -s ${cuda_base_path}/lib ${cuda_base_path}/lib64
rm -rf ${cuda_base_path}/targets

set +x
dpkg -l | grep cudnn
mkdir -p "${prefix}"

for cudnn_file in $(dpkg -L libcudnn${CUDNN_MAJOR_VERSION}-cuda-${CUDA_MAJOR_VERSION} libcudnn${CUDNN_MAJOR_VERSION}-dev-cuda-${CUDA_MAJOR_VERSION} | sort -u); do
  if [[ -f "${cudnn_file}" || -h "${cudnn_file}" ]]; then
    nosysprefix="${cudnn_file#"/usr/"}"
    noarchinclude="${nosysprefix/#"include/${arch}"/include}"
    noverheader="${noarchinclude/%"_v${CUDNN_MAJOR_VERSION}.h"/.h}"
    noarchlib="${noverheader/#"lib/${arch}"/lib}"
    
    if [[ -d "${prefix}" ]]; then
      link_name="${prefix}/${noarchlib}"
    else
      link_name="/usr/local/cuda/lib64/${noarchlib}"
    fi
    
    link_dir=$(dirname "${link_name}")
    mkdir -p "${link_dir}"
    echo "linking ${cudnn_file} -> ${link_name}"
    ln -s "${cudnn_file}" "${link_name}"
  fi
done
