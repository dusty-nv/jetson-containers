# openvla

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`openvla`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`bitsandbytes`](/packages/llm/bitsandbytes) [`tensorboard`](/packages/ml/tensorflow/tensorboard) [`h5py`](/packages/build/h5py) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/openvla:r36.3.0`](https://hub.docker.com/r/dustynv/openvla/tags) `(2024-08-26, 6.5GB)`<br>[`dustynv/openvla:r36.4.3-cu128-cp312-24.04`](https://hub.docker.com/r/dustynv/openvla/tags) `(2025-02-27, 6.7GB)` |

| **`openvla:mimicgen`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`bitsandbytes`](/packages/llm/bitsandbytes) [`tensorboard`](/packages/ml/tensorflow/tensorboard) [`h5py`](/packages/build/h5py) [`opengl`](/packages/multimedia/opengl) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`numba`](/packages/numeric/numba) [`mujoco`](/packages/sim/mujoco) [`robosuite`](/packages/sim/robosuite) [`robomimic`](/packages/sim/robomimic) [`mimicgen`](/packages/sim/mimicgen) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/openvla:r36.3.0`](https://hub.docker.com/r/dustynv/openvla/tags) | `2024-08-26` | `arm64` | `6.5GB` |
| &nbsp;&nbsp;[`dustynv/openvla:r36.4.3-cu128-cp312-24.04`](https://hub.docker.com/r/dustynv/openvla/tags) | `2025-02-27` | `arm64` | `6.7GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag openvla)

# or explicitly specify one of the container images above
jetson-containers run dustynv/openvla:r36.4.3-cu128-cp312-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/openvla:r36.4.3-cu128-cp312-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag openvla)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag openvla) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build openvla
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
