#!/usr/bin/env python3

import multiprocessing as mp
import sys, os


def worker(hook_mode=None):
    import logging, time, gc
    import torch
    from torch_memory_saver import torch_memory_saver as tms
    from torch_memory_saver.testing_utils import get_and_print_gpu_memory

    if hook_mode is not None:
        tms.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    big = 1_000_000_000 if device == "cuda" else 10_000_000

    normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device=device)
    with tms.region():
        pauseable_tensor = torch.full((big,), 100, dtype=torch.uint8, device=device)

    get_and_print_gpu_memory("Before pause");
    time.sleep(0.5)
    tms.pause();
    get_and_print_gpu_memory("After pause");
    time.sleep(0.5)
    tms.resume();
    get_and_print_gpu_memory("After resume")

    # explicit teardown
    pauseable_tensor = None
    normal_tensor = None
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()

    # avoid shutdown-time GC surprises
    gc.disable();
    gc.collect();
    gc.enable()

    # HARD EXIT from child (skips interpreter teardown)
    os._exit(0)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    hook_mode = "torch"
    p = mp.Process(target=worker, args=(hook_mode,))
    p.start();
    p.join()
    sys.exit(p.exitcode)

print('torch-memory-saver OK')
