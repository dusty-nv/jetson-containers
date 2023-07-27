# l4t-ml

|            |            |
|------------|------------|
| Builds | [![`l4t-ml_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-ml_jp46.yml?label=l4t-ml_jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-ml_jp46.yml) [![`l4t-ml_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-ml_jp51.yml?label=l4t-ml_jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-ml_jp51.yml) |
| Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) [`tensorflow2`](/packages/tensorflow) [`opencv`](/packages/opencv) [`pycuda`](/packages/pycuda) [`cupy`](/packages/cupy) [`onnxruntime`](/packages/onnxruntime) [`numba`](/packages/numba) [`rust`](/packages/rust) [`jupyterlab`](/packages/jupyterlab) |
| Dockerfile | [`Dockerfile`](Dockerfile) |
### Container Images
- [`dustynv/l4t-ml:r32.6.1-py3`](https://hub.docker.com/r/dustynv/l4t-ml/tags)  `arm64`  `(1.5GB)`
- [`dustynv/l4t-ml:r32.5.0-py3`](https://hub.docker.com/r/dustynv/l4t-ml/tags)  `arm64`  `(1.7GB)`

### Run Container
The [`run.sh`](/run.sh) script sets some default `docker run` args like `--runtime nvidia` and auto-mounts some devices.  It prints out the full `docker run` command generated.
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag l4t-ml)

# or manually specify one of the container images above
./run.sh dustynv/l4t-ml:r32.6.1-py3
```
To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag l4t-ml)
```
To start the container running a command as opposed to the shell:
```bash
./run.sh $(./autotag l4t-ml) my_app --abc xyz
```
