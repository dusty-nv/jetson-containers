## Holoscan

This package provides containers for [Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk). The Holoscan SDK can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, Industrial Inspection and more.

> ⚠️ Note: This package installs the Holoscan SDK python wheel, see the [support matrix](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#not-sure-what-to-choose) to understand which Holoscan features this enables.

## To run a sample Holoscan app:
```bash
python3 /opt/nvidia/holoscan-sdk/examples/hello_world/python/hello_world.py
 ```
Example output:
```bash
[info] [gxf_executor.cpp:248] Creating context
[info] [gxf_executor.cpp:1691] Loading extensions from configs...
[info] [gxf_executor.cpp:1897] Activating Graph...
[info] [gxf_executor.cpp:1929] Running Graph...
[info] [gxf_executor.cpp:1931] Waiting for completion...
2024-06-10 22:41:50.358 INFO  gxf/std/greedy_scheduler.cpp@191: Scheduling 1 entities

Hello World!

2024-06-10 22:41:50.359 INFO  gxf/std/greedy_scheduler.cpp@338: Scheduler stopped: no more entities to schedule
2024-06-10 22:41:50.359 INFO  gxf/std/greedy_scheduler.cpp@401: Scheduler finished.
[info] [gxf_executor.cpp:1934] Deactivating Graph...
[info] [gxf_executor.cpp:1942] Graph execution finished.
[info] [gxf_executor.cpp:276] Destroying context
```
