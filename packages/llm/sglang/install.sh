#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of sglang ${SGLANG_VERSION}"
	exit 1
fi
pip3 install \
  compressed-tensors \
  datasets \
  decord \
  fastapi \
  hf_transfer \
  huggingface_hub \
  interegular \
  "llguidance>=0.7.11,<0.8.0" \
  modelscope \
  ninja \
  orjson \
  packaging \
  partial_json_parser \
  pillow \
  "prometheus-client>=0.20.0" \
  psutil \
  pydantic \
  pynvml \
  python-multipart \
  "pyzmq>=25.1.2" \
  "soundfile>=0.13.1" \
  "torchao>=0.9.0" \
  "transformers==4.51.1" \
  uvicorn \
  uvloop \
  "blobfile>=3.0.0"

pip3 install sgl-kernel "sglang[all]~=${SGLANG_VERSION}" || \
pip3 install sgl-kernel "sglang[all]~=${SGLANG_VERSION_SPEC}"