#!/usr/bin/env bash

set -exo pipefail

echo "Testing wyoming-piper..."

python3 -c 'import wyoming_piper; print(wyoming_piper.__version__);'

echo "wyoming-piper OK"
