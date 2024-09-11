#!/bin/bash

set -e

CUDNN_MAJOR_VERSION=9
prefix=/opt/nvidia/cudnn
arch=$(uname -m)-linux-gnu


for cudnn_file in $(dpkg -L libcudnn${CUDNN_MAJOR_VERSION} libcudnn${CUDNN_MAJOR_VERSION}-dev | sort -u); do
  if [[ -f "${cudnn_file}" || -h "${cudnn_file}" ]]; then
    nosysprefix="${cudnn_file#"/usr/"}"
    noarchinclude="${nosysprefix/#"include/${arch}"/include}"
    noverheader="${noarchinclude/%"_v${CUDNN_MAJOR_VERSION}.h"/.h}"
    noarchlib="${noverheader/#"lib/${arch}"/lib}"
    link_name="${prefix}/${noarchlib}"
    
    link_dir=$(dirname "${link_name}")
    mkdir -p "${link_dir}"
    ln -s "${cudnn_file}" "${link_name}"
  fi
done

