## Holoscan

This package provides containers for [Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk) and running sample apps from [HoloHub](https://github.com/nvidia-holoscan/holohub). The Holoscan SDK can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, Industrial Inspection and more.

## TLDR;
To run a sample Holoscan app:
```bash
jetson-containers run \
 # Mount Holohub repo directly to this folder
 -v <jetson-containers_dir>/packages/holoscan-jetson/holohub:/opt/nvidia/holohub
 # Use autotag
 $(autotag holoscan-jetson) \
 # Build and run a HoloHub sample app
 /bin/bash -c \
 './run build endoscopy_tool_tracking \
 && ./run launch endoscopy_tool_tracking
 ```

## Running options:
> ⚠️ Note: Holoscan SDK defaults to downloading models to  `<holohub_dir>/data` and build artifacts to `<holohub_dir>/build`. Without mounting these to a docker volume, you will have to re-build and re-download data each run.

To remedy this, you have several options:
1. Mount all of HoloHub to this package folder:
```bash
jetson-containers run \
 # Mount Holohub repo directly to this folder
 -v <jetson-containers_dir>/packages/holoscan-jetson/holohub:/opt/nvidia/holohub
 # Use autotag
 $(autotag holoscan-jetson)
 ```
 2. Mount only the `/data` and `/build` dirs to this package folder:
```bash
jetson-containers run \
 # Mount HoloHub /build and /data dirs
 -v <jetson-containers_dir>/packages/holoscan-jetson/data:/opt/nvidia/holohub/data \
 -v <jetson-containers_dir>/packages/holoscan-jetson/build:/opt/nvidia/holohub/build \
 # Use autotag
 $(autotag holoscan-jetson)
 ```
  3. Place the `/data` and `/build` dirs to the default `jetson-containers` `/data` mount and use the `--buildpath` to redirect to this location:
```bash
jetson-containers run $(autotag holoscan-jetson) /bin/bash -c \
  # specify the --buildpath for building and running apps
 './run build endoscopy_tool_tracking --buildpath /data/holoscan/build \
 && ./run launch endoscopy_tool_tracking --buildpath /data/holoscan/build
 ```
