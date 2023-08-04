# protobuf_cpp

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`protobuf:cpp`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `protobuf` |
| &nbsp;&nbsp;&nbsp;Builds | [![`protobuf-cpp_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/protobuf-cpp_jp51.yml?label=protobuf-cpp:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/protobuf-cpp_jp51.yml) [![`protobuf-cpp_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/protobuf-cpp_jp46.yml?label=protobuf-cpp:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/protobuf-cpp_jp46.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) [`tensorflow`](/packages/tensorflow) [`tensorflow2`](/packages/tensorflow) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/protobuf:cpp-r32.7.1`](https://hub.docker.com/r/dustynv/protobuf/tags) `(2023-07-29, 0.4GB)`<br>[`dustynv/protobuf:cpp-r35.3.1`](https://hub.docker.com/r/dustynv/protobuf/tags) `(2023-07-29, 5.1GB)` |
| &nbsp;&nbsp;&nbsp;Notes | build protobuf using cpp implementation (https://jkjung-avt.github.io/tf-trt-revisited/) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag protobuf_cpp)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host protobuf_cpp:35.2.1

```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag protobuf_cpp)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag protobuf_cpp) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh protobuf_cpp
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
