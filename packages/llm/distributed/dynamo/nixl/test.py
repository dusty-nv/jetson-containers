#!/usr/bin/env python3
print('testing nixl...')
import time

import nixl._utils as nixl_utils
from nixl._api import nixl_agent

if __name__ == "__main__":
    desc_count = 24 * 64 * 1024
    agent = nixl_agent("test", None)
    addr = nixl_utils.malloc_passthru(256)

    addr_list = [(addr, 256, 0)] * desc_count

    start_time = time.perf_counter()

    descs = agent.get_xfer_descs(addr_list, "DRAM", True)

    end_time = time.perf_counter()

    assert descs.descCount() == desc_count

    print(
        "Time per desc add in us:", (1000000.0 * (end_time - start_time)) / desc_count
    )
    nixl_utils.free_passthru(addr)
print('nixl OK\n')