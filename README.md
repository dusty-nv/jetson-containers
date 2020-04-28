# Machine Learning Containers for Jetson and JetPack 4.4

Hosted on [NVIDIA GPU Cloud](https://ngc.nvidia.com/catalog/containers?orderBy=modifiedDESC&query=L4T&quickFilter=containers&filters=) (NGC) are the following Docker container images for machine learning on Jetson:

* [`l4t-ml`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-ml)
* [`l4t-pytorch`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch)
* [`l4t-tensorflow`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow)

Included in this repo are the Dockerfiles and scripts used to build the above containers.

## Building the Containers

To rebuild the containers from a Jetson device running [JetPack 4.4 Developer Preview](https://developer.nvidia.com/embedded/jetpack), clone this repo and run `./scripts/docker_build_all.sh`:

``` bash
$ git clone https://github.com/dusty-nv/jetson-containers
$ cd jetson-containers
$ ./scripts/docker_build_all.sh
```

Note that the TensorFlow and PyTorch pip wheel installers for aarch64 are automatically downloaded in the Dockerfiles from the [Jetson Zoo](https://elinux.org/Jetson_Zoo).

## Testing the Containers

To run a series of automated tests on the packages installed in the containers, run the following from your `jetson-containers` directory:

``` bash
$ ./scripts/docker_test.sh
```
