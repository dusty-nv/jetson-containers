## Holoscan

This package provides containers for [Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk). The Holoscan SDK can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, Industrial Inspection and more.

> ⚠️ Note: This package installs the Holoscan SDK Debian package, see the [support matrix](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#not-sure-what-to-choose) to understand which Holoscan SDK features this enables.

## To run a sample Holoscan-SDK app:
```bash
export PYTHONPATH=$PYTHONPATH:/opt/nvidia/holoscan/python/lib
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

## Running HoloHub Apps:
[HoloHub](https://github.com/nvidia-holoscan/holohub/tree/main) is a central repository for the NVIDIA Holoscan AI sensor processing community to share apps and extensions.

Here is an example of how to run the [endoscopy_tool_tracking](https://github.com/nvidia-holoscan/holohub/tree/main/applications/endoscopy_tool_tracking) example:
```bash
export HOLOHUB_BUILD_PATH=/data/holohub/endoscopy_tool_tracking
export PYTHONPATH=$PYTHONPATH:$HOLOHUB_BUILD_PATH/python/lib:/opt/nvidia/holoscan/python/lib
cd /opt/nvidia/holohub
./run build endoscopy_tool_tracking --buildpath $HOLOHUB_BUILD_PATH
cd $HOLOHUB_BUILD_PATH
python3 /opt/nvidia/holohub/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.python --data /opt/nvidia/holohub/data/endoscopy/
```

These apps are often more complex than the examples included in the Holoscan-SDK. As such, many apps will require that you build them first with CMake, regardless of whether you are using Python or C++. Note the use of the `--buildpath` arg, this redirects the build location from the default location to the `/data` volume that is mounted by the Jetson-Containers repo. This ensures that builds will persist across Docker runs.

