# l4t-tensorflow

<details open>
<summary><h3>l4t-tensorflow:tf1</h3></summary>

|            |            |
|------------|------------|
| Builds | [![`l4t-tensorflow-tf1_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf1_jp46.yml?label=l4t-tensorflow-tf1_jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf1_jp46.yml) [![`l4t-tensorflow-tf1_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf1_jp51.yml?label=l4t-tensorflow-tf1_jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf1_jp51.yml) |
| Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) [`tensorflow`](/packages/tensorflow) [`opencv`](/packages/opencv) [`pycuda`](/packages/pycuda) |
</details>
<details open>
<summary><h3>l4t-tensorflow:tf2</h3></summary>

|            |            |
|------------|------------|
| Builds | [![`l4t-tensorflow-tf2_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf2_jp46.yml?label=l4t-tensorflow-tf2_jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf2_jp46.yml) [![`l4t-tensorflow-tf2_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf2_jp51.yml?label=l4t-tensorflow-tf2_jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf2_jp51.yml) |
| Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) [`tensorflow2`](/packages/tensorflow) [`opencv`](/packages/opencv) [`pycuda`](/packages/pycuda) |
</details>
### Run Container
The [`run.sh`](/run.sh) script sets some default `docker run` args like `--runtime nvidia` and auto-mounts some devices.  It prints out the full `docker run` command generated.
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag l4t-tensorflow)
```
To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag l4t-tensorflow)
```
To start the container running a command as opposed to the shell:
```bash
./run.sh $(./autotag l4t-tensorflow) my_app --abc xyz
```
