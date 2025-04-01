#!/usr/bin/env bash

#cd /opt/llm-awq

python3 -c 'import awq'
python3 -c 'import tinychat'

python3 -m awq.entry --help
