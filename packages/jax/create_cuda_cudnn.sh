#!/bin/bash

set -e

CUDNN_MAJOR_VERSION=9
CUDA_MAJOR_VERSION=12.2
prefix=/opt/nvidia/cudnn
arch=$(uname -m)-linux-gnu
cuda_base_path="/usr/local/cuda-${CUDA_MAJOR_VERSION}"

# Comprobar si la ruta especificada de CUDA existe
if [[ -d "${cuda_base_path}" ]]; then
  cuda_lib_path="${cuda_base_path}/lib64"
  output_path="/usr/local/cuda-${CUDA_MAJOR_VERSION}/lib"
else
  cuda_lib_path="/usr/local/cuda/lib64"
  output_path="/usr/local/cuda/lib64"
fi

# Crear enlace simbólico para CUDA
sudo ln -s "${cuda_lib_path}" "${output_path}"

# Proceso para CUDNN
for cudnn_file in $(dpkg -L libcudnn${CUDNN_MAJOR_VERSION} libcudnn${CUDNN_MAJOR_VERSION}-dev | sort -u); do
  if [[ -f "${cudnn_file}" || -h "${cudnn_file}" ]]; then
    nosysprefix="${cudnn_file#"/usr/"}"
    noarchinclude="${nosysprefix/#"include/${arch}"/include}"
    noverheader="${noarchinclude/%"_v${CUDNN_MAJOR_VERSION}.h"/.h}"
    noarchlib="${noverheader/#"lib/${arch}"/lib}"
    
    # Usar la ruta cuda_base_path o /usr/local/cuda/lib64
    if [[ -d "${cuda_base_path}" ]]; then
      link_name="${cuda_base_path}/${noarchlib}"
    else
      link_name="/usr/local/cuda/lib64/${noarchlib}"
    fi
    
    link_dir=$(dirname "${link_name}")
    mkdir -p "${link_dir}"
    ln -s "${cudnn_file}" "${link_name}"
  fi
done
