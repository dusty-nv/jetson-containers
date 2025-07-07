# vllm

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`vllm:0.7.4`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`torchaudio`](/packages/pytorch/torchaudio) [`ninja`](/packages/build/ninja) [`causalconv1d`](/packages/ml/mamba/causalconv1d) [`mamba`](/packages/ml/mamba/mamba) [`xgrammar`](/packages/llm/xgrammar) [`flashinfer`](/packages/attention/flash-infer) [`ffmpeg`](/packages/multimedia/ffmpeg) [`torchao`](/packages/pytorch/torchao) [`flexprefill`](/packages/attention/flexprefill) [`minference`](/packages/llm/minference) [`torch-memory-saver`](/packages/pytorch/torchsaver) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/vllm:0.7.4-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/vllm/tags) `(2025-03-17, 5.2GB)` |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/vllm-project/vllm |

| **`vllm:0.8.4`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`torchaudio`](/packages/pytorch/torchaudio) [`ninja`](/packages/build/ninja) [`causalconv1d`](/packages/ml/mamba/causalconv1d) [`mamba`](/packages/ml/mamba/mamba) [`xgrammar`](/packages/llm/xgrammar) [`flashinfer:0.2.1.post2`](/packages/attention/flash-infer) [`ffmpeg`](/packages/multimedia/ffmpeg) [`torchao`](/packages/pytorch/torchao) [`flexprefill`](/packages/attention/flexprefill) [`minference`](/packages/llm/minference) [`torch-memory-saver`](/packages/pytorch/torchsaver) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/vllm-project/vllm |

| **`vllm:v0.8.5.post1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`torchaudio`](/packages/pytorch/torchaudio) [`ninja`](/packages/build/ninja) [`causalconv1d`](/packages/ml/mamba/causalconv1d) [`mamba`](/packages/ml/mamba/mamba) [`xgrammar`](/packages/llm/xgrammar) [`flashinfer:0.2.2.post1`](/packages/attention/flash-infer) [`ffmpeg`](/packages/multimedia/ffmpeg) [`torchao`](/packages/pytorch/torchao) [`flexprefill`](/packages/attention/flexprefill) [`minference`](/packages/llm/minference) [`torch-memory-saver`](/packages/pytorch/torchsaver) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/vllm-project/vllm |

| **`vllm:0.9.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`torchaudio`](/packages/pytorch/torchaudio) [`ninja`](/packages/build/ninja) [`causalconv1d`](/packages/ml/mamba/causalconv1d) [`mamba`](/packages/ml/mamba/mamba) [`xgrammar`](/packages/llm/xgrammar) [`flashinfer`](/packages/attention/flash-infer) [`ffmpeg`](/packages/multimedia/ffmpeg) [`torchao`](/packages/pytorch/torchao) [`flexprefill`](/packages/attention/flexprefill) [`minference`](/packages/llm/minference) [`torch-memory-saver`](/packages/pytorch/torchsaver) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/vllm-project/vllm |

| **`vllm:0.9.2`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`torchaudio`](/packages/pytorch/torchaudio) [`ninja`](/packages/build/ninja) [`causalconv1d`](/packages/ml/mamba/causalconv1d) [`mamba`](/packages/ml/mamba/mamba) [`xgrammar`](/packages/llm/xgrammar) [`flashinfer`](/packages/attention/flash-infer) [`ffmpeg`](/packages/multimedia/ffmpeg) [`torchao`](/packages/pytorch/torchao) [`flexprefill`](/packages/attention/flexprefill) [`minference`](/packages/llm/minference) [`torch-memory-saver`](/packages/pytorch/torchsaver) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/vllm-project/vllm |

| **`vllm:0.9.3`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `vllm` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`torchaudio`](/packages/pytorch/torchaudio) [`ninja`](/packages/build/ninja) [`causalconv1d`](/packages/ml/mamba/causalconv1d) [`mamba`](/packages/ml/mamba/mamba) [`xgrammar`](/packages/llm/xgrammar) [`flashinfer`](/packages/attention/flash-infer) [`ffmpeg`](/packages/multimedia/ffmpeg) [`torchao`](/packages/pytorch/torchao) [`flexprefill`](/packages/attention/flexprefill) [`minference`](/packages/llm/minference) [`torch-memory-saver`](/packages/pytorch/torchsaver) |
| &nbsp;&nbsp;&nbsp;Dependants | [`cosmos-reason1`](/packages/diffusion/cosmos/cosmos-reason1) [`dynamo:0.3.2`](/packages/llm/dynamo/dynamo) [`l4t-dynamo`](/packages/ml/l4t/l4t-dynamo) [`llama-factory`](/packages/llm/llama-factory) [`sglang:0.4.4`](/packages/llm/sglang) [`sglang:0.4.6`](/packages/llm/sglang) [`sglang:0.4.9`](/packages/llm/sglang) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/vllm-project/vllm |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/vllm:0.6.3-r36.4.0`](https://hub.docker.com/r/dustynv/vllm/tags) | `2024-10-21` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/vllm:0.6.6.post1-builder-r36.4.0`](https://hub.docker.com/r/dustynv/vllm/tags) | `2025-01-15` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/vllm:0.6.6.post1-r36.4.0`](https://hub.docker.com/r/dustynv/vllm/tags) | `2025-01-17` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/vllm:0.7.4-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/vllm/tags) | `2025-03-17` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/vllm:0.8.3-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/vllm/tags) | `2025-04-02` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/vllm:0.8.5-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/vllm/tags) | `2025-04-22` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/vllm:0.8.6-r36.4-cu128-24.04`](https://hub.docker.com/r/dustynv/vllm/tags) | `2025-05-06` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/vllm:0.8.6-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/vllm/tags) | `2025-05-01` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/vllm:0.8.6-sbsa-cu128`](https://hub.docker.com/r/dustynv/vllm/tags) | `2025-05-01` | `arm64` | `10.9GB` |
| &nbsp;&nbsp;[`dustynv/vllm:r36.4.0`](https://hub.docker.com/r/dustynv/vllm/tags) | `2024-10-15` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/vllm:r36.4.0-cu128`](https://hub.docker.com/r/dustynv/vllm/tags) | `2025-02-10` | `arm64` | `4.5GB` |

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
jetson-containers run $(autotag vllm)

# or explicitly specify one of the container images above
jetson-containers run dustynv/vllm:0.8.6-r36.4-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/vllm:0.8.6-r36.4-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag vllm)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag vllm) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build vllm
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
