#!/usr/bin/env bash

source /workspace/speaches/.venv/bin/activate

echo "=== Testing speaches import ==="
python3 -c 'import speaches; print("✓ speaches imported successfully")' || { echo "✗ Failed to import speaches"; exit 1; }

echo ""
echo "=== Testing CTranslate2 import ==="
python3 -c 'import ctranslate2; print("✓ CTranslate2 version:", ctranslate2.__version__)' || { echo "✗ Failed to import ctranslate2"; exit 1; }

echo ""
echo "=== Testing CUDA device detection ==="
python3 -c '
import ctranslate2 as ct
device_count = ct.get_cuda_device_count()
print(f"CUDA devices detected: {device_count}")
if device_count != 1:
    print(f"✗ CTranslate2 was not built with CUDA support")
    exit(1)
else:
    print("✓ CUDA device detection successful")
' || { echo "✗ CUDA device detection failed"; exit 1; }

echo ""
echo "=== All tests passed ✓ ===" 