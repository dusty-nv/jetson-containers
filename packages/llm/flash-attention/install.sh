#!/usr/bin/env bash
set -ex

pip3 install --no-cache-dir --verbose flash-attn==${FLASH_ATTENTION_VERSION}
pip3 show flash-attn && python3 -c 'import flash_attn'