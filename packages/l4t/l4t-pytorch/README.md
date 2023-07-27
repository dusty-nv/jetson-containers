# l4t-pytorch

<details open>
<summary><b>CONTAINERS</b></summary>
</br>

| **`l4t-pytorch`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`l4t-pytorch_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp46.yml?label=l4t-pytorch_jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-pytorch_jp46.yml) [![`l4t-pytorch_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp51.yml?label=l4t-pytorch_jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-pytorch_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) [`torch2trt`](/packages/pytorch/torch2trt) [`opencv`](/packages/opencv) [`pycuda`](/packages/pycuda) |

</details>

<details open>
<summary><b>CONTAINER IMAGES</b></summary>
</br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/l4t-pytorch:r35.1.0-pth1.11-py3`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) | `2022-09-20` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/l4t-pytorch:r35.1.0-pth1.12-py3`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) | `2022-09-20` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/l4t-pytorch:r35.1.0-pth1.13-py3`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) | `2022-09-20` | `arm64` | `5.8GB` |

Container images are compatible with other minor versions of JetPack/L4T:
* L4T R32.7.1 containers can run on other versions of L4T R32.7 (JetPack 4.6+)
* L4T R35.2.1 containers can run on other versions of L4T R35.x (JetPack 5.1+)
</details>

<details open>
<summary><b>RUN CONTAINER</b></summary>
</br>

[`run.sh`](/run.sh) adds some default `docker run` args (like `--runtime nvidia`, mounts a [`/data`](/data) cache, and detects devices)
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag l4t-pytorch)

# or manually specify one of the container images above
./run.sh dustynv/l4t-pytorch:r35.1.0-pth1.13-py3

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/l4t-pytorch:r35.1.0-pth1.13-py3
```
To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag l4t-pytorch)
```
To start the container running a command, as opposed to the shell:
```bash
./run.sh $(./autotag l4t-pytorch) my_app --abc xyz
```
</details>
<details open>
<summary><b>BUILD CONTAINER</b></summary>
</br>

If you use [`autotag`](/autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do this System Setup, then run:
```bash
./build.sh l4t-pytorch
```
The dependencies from above will be built into the container, and it'll be tested.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
