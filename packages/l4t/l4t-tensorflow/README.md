# l4t-tensorflow

<details open>
<summary><b>CONTAINERS</b></summary>
<br>

| **`l4t-tensorflow:tf1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`l4t-tensorflow-tf1_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf1_jp46.yml?label=l4t-tensorflow-tf1_jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf1_jp46.yml) [![`l4t-tensorflow-tf1_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf1_jp51.yml?label=l4t-tensorflow-tf1_jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf1_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) [`tensorflow`](/packages/tensorflow) [`opencv`](/packages/opencv) [`pycuda`](/packages/pycuda) |

| **`l4t-tensorflow:tf2`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`l4t-tensorflow-tf2_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf2_jp46.yml?label=l4t-tensorflow-tf2_jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf2_jp46.yml) [![`l4t-tensorflow-tf2_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf2_jp51.yml?label=l4t-tensorflow-tf2_jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf2_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) [`tensorflow2`](/packages/tensorflow) [`opencv`](/packages/opencv) [`pycuda`](/packages/pycuda) |

</details>

<details open>
<summary><b>RUN CONTAINER</b></summary>
<br>

To start the container, you can use the [`run.sh`](/run.sh)/[`autotag`](/autotag) helpers or construct a full [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag l4t-tensorflow)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host l4t-tensorflow:35.2.1

```
> <sup>[`run.sh`](/run.sh) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from DockerHub, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag l4t-tensorflow)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag l4t-tensorflow) my_app --abc xyz
```
You can pass any arguments to `run.sh` that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it will print out the full run command that it constructs before executing it.
</details>
<details open>
<summary><b>BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do this System Setup, then run:
```bash
./build.sh l4t-tensorflow
```
The dependencies from above will be built into the container, and it'll be tested.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
