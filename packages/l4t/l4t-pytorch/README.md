# l4t-pytorch

|            |            |
|------------|------------|
| Builds | [![`l4t-pytorch_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp46.yml?label=l4t-pytorch_jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-pytorch_jp46.yml) [![`l4t-pytorch_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp51.yml?label=l4t-pytorch_jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-pytorch_jp51.yml) |
| Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) [`opencv`](/packages/opencv) [`pycuda`](/packages/pycuda) |
### Container Images
- [`dustynv/l4t-pytorch:r35.1.0-pth1.13-py3`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags)  `arm64`  `(5.8GB)`
- [`dustynv/l4t-pytorch:r35.1.0-pth1.12-py3`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags)  `arm64`  `(5.8GB)`
- [`dustynv/l4t-pytorch:r35.1.0-pth1.11-py3`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags)  `arm64`  `(5.7GB)`

### Run Container
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag l4t-pytorch)

# or manually specify one of the container images above:
./run.sh `dustynv/l4t-pytorch:r35.1.0-pth1.13-py3`
```
To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag l4t-pytorch)
```
