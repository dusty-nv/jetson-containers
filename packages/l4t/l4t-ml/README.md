# l4t-ml

<details open>
<summary><b>CONTAINERS</b></summary>
</br>

| **`l4t-ml`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`l4t-ml_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-ml_jp46.yml?label=l4t-ml_jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-ml_jp46.yml) [![`l4t-ml_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-ml_jp51.yml?label=l4t-ml_jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-ml_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) [`tensorflow2`](/packages/tensorflow) [`opencv`](/packages/opencv) [`pycuda`](/packages/pycuda) [`cupy`](/packages/cupy) [`onnxruntime`](/packages/onnxruntime) [`numba`](/packages/numba) [`rust`](/packages/rust) [`jupyterlab`](/packages/jupyterlab) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b>CONTAINER IMAGES</b></summary>
</br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/l4t-ml:r32.5.0-py3`](https://hub.docker.com/r/dustynv/l4t-ml/tags) | `2021-06-29` | `arm64` | `1.7GB` |
| &nbsp;&nbsp;[`dustynv/l4t-ml:r32.6.1-py3`](https://hub.docker.com/r/dustynv/l4t-ml/tags) | `2021-12-13` | `arm64` | `1.5GB` |

<sub>Container images are compatible with other minor versions of JetPack/L4T:
* L4T R32.7.1 containers can run on other versions of L4T R32.7 (JetPack 4.6+)
* L4T R35.2.1 containers can run on other versions of L4T R35.x (JetPack 5.1+)
</sub></details>

<details open>
<summary><b>RUN CONTAINER</b></summary>
</br>

[`run.sh`](/run.sh) adds some default `docker run` args (like `--runtime nvidia`, mounts a [`/data`](/data) cache, and detects devices)
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag l4t-ml)

# or manually specify one of the container images above
./run.sh dustynv/l4t-ml:r32.6.1-py3

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/l4t-ml:r32.6.1-py3
```
To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag l4t-ml)
```
To start the container running a command, as opposed to the shell:
```bash
./run.sh $(./autotag l4t-ml) my_app --abc xyz
```
</details>
<details open>
<summary><b>BUILD CONTAINER</b></summary>
</br>

If you use [`autotag`](/autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do this System Setup, then run:
```bash
./build.sh l4t-ml
```
The dependencies from above will be built into the container, and it'll be tested.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
