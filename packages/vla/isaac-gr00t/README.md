# Isaac-GR00T

The [`jetson-containers build`](/jetson-containers) command is a proxy launcher for [`jetson_containers/build.py`](/jetson_containers/build.py).  It can be run from any working directory after you clone the repo and run the installer from the [System Setup](/docs/setup.md) (you should also probably mount additional storage when building containers)

To list the packages available to build for your version of JetPack/L4T, you can use `--list-packages` and `--show-packages`:

```bash
$ jetson-containers build --list-packages       # list all packages
$ jetson-containers build --show-packages       # show all package metadata
$ jetson-containers build --show-packages ros*  # show all the ros packages
```

## Build Container

To build a container for Isaac-GR00T:

```bash
$ jetson-containers build isaac-gr00t
```

## Run Container

To run the container:

```bash
docker run -it --rm --network=host --runtime=nvidia --volume /mnt:/mnt --workdir /mnt/Isaac-GR00T   isaac-groot:r36.4-cu126-22.04     /bin/bash
```

## Install nsys

To install nsys in the container(below is example for JetPack 6.2):

```bash
echo -e "deb https://repo.download.nvidia.com/jetson/common r36.4 main\ndeb https://repo.download.nvidia.com/jetson/t234 r36.4 main\ndeb https://repo.download.nvidia.com/jetson/ffmpeg r36.4 main" | tee /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
apt-key adv --fetch-keys https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
apt update
apt list --upgradable
apt install -y nsight-systems-2024.5.4
```

## Run nsys profile

To run nsys profile:

```bash
nsys profile -t cuda,nvtx,tegra-accelerators,osrt --show-output=true --force-overwrite=true --output=%p python3 gr00t_inference.py
```

