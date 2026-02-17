#!/usr/bin/env bash

python3 -m torch.utils.collect_env

# Test torch.distributed functionality
python3 -c "
import torch
import torch.distributed as dist
import sys

def test_torch_distributed():
    \"\"\"Test torch.distributed functionality with proper checks and status output\"\"\"
    print('\\n=== Testing torch.distributed ===')
    is_available = False

    # Check if distributed is available
    try:
        is_available = dist.is_available()
        if not is_available:
            print('❌ torch.distributed is not available')
            return 1
        print('✅ torch.distributed is available')
    except Exception as e:
        print(f'❌ Error checking availability: {e}')
        return 1

    # Check if distributed is initialized
    try:
        is_initialized = dist.is_initialized()
        if is_available and not is_initialized:
            print('ℹ️  torch.distributed is available but not initialized - this is normal for single-process testing')
            return 0
        elif is_initialized:
            print('✅ torch.distributed is initialized')
        else:
            print('❌ torch.distributed initialization check failed')
            return 1
    except Exception as e:
        print(f'❌ Error checking initialization: {e}')
        return 1

    # If we reach here, distributed is both available and initialized
    # Run all inner checks

    # Check backend
    try:
        backend = dist.get_backend()
        print(f'✅ Backend: {backend}')
    except Exception as e:
        print(f'❌ Error getting backend: {e}')
        return 1

    # Check world size
    try:
        world_size = dist.get_world_size()
        if world_size > 0:
            print(f'✅ World size: {world_size}')
        else:
            print(f'❌ Invalid world size: {world_size}')
            return 1
    except Exception as e:
        print(f'❌ Error getting world size: {e}')
        return 1

    # Check rank
    try:
        rank = dist.get_rank()
        if rank >= 0:
            print(f'✅ Rank: {rank}')
        else:
            print(f'❌ Invalid rank: {rank}')
            return 1
    except Exception as e:
        print(f'❌ Error getting rank: {e}')
        return 1

    print('✅ All torch.distributed checks passed')
    return 0

# Run the test function
exit_code = test_torch_distributed()
print('=== torch.distributed tests completed ===\\n')
sys.exit(exit_code)
"

if [[ ${ENABLE_DISTRIBUTED_JETSON_NCCL:-0} != "1" ]]; then
    echo "Skipping distributed NCCL test for Jetson, to enable build cudastack:distributed"
    exit 0
fi

torchrun --nproc-per-node=1  /test/distributed_test.py
