#!/usr/bin/env bash

echo "Testing wyoming-openwakeword..."

python3 -c 'import wyoming_openwakeword; print(wyoming_openwakeword.__version__);'

echo "wyoming-openwakeword OK"
