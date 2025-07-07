# sglang

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`sglang:0.4.4`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`flashinfer`](/packages/attention/flash-infer) [`torchao`](/packages/pytorch/torchao) [`cuda-python`](/packages/cuda/cuda-python) [`ffmpeg`](/packages/multimedia/ffmpeg) [`ninja`](/packages/build/ninja) [`torchaudio`](/packages/pytorch/torchaudio) [`causalconv1d`](/packages/ml/mamba/causalconv1d) [`mamba`](/packages/ml/mamba/mamba) [`xgrammar`](/packages/llm/xgrammar) [`flashinfer`](/packages/attention/flash-infer) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`flexprefill`](/packages/attention/flexprefill) [`minference`](/packages/llm/minference) [`torch-memory-saver`](/packages/pytorch/torchsaver) [`vllm`](/packages/llm/vllm) [`genai-bench`](/packages/llm/sglang/genai-bench) [`sgl-kernel`](/packages/llm/sglang/sgl-kernel) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/sglang:0.4.4-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/sglang/tags) `(2025-03-03, 5.4GB)` |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/sgl-project/sglang |

| **`sglang:0.4.6`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`flashinfer:0.2.6.post1`](/packages/attention/flash-infer) [`torchao`](/packages/pytorch/torchao) [`cuda-python`](/packages/cuda/cuda-python) [`ffmpeg`](/packages/multimedia/ffmpeg) [`ninja`](/packages/build/ninja) [`torchaudio`](/packages/pytorch/torchaudio) [`causalconv1d`](/packages/ml/mamba/causalconv1d) [`mamba`](/packages/ml/mamba/mamba) [`xgrammar`](/packages/llm/xgrammar) [`flashinfer:0.2.6.post1`](/packages/attention/flash-infer) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`flexprefill`](/packages/attention/flexprefill) [`minference`](/packages/llm/minference) [`torch-memory-saver`](/packages/pytorch/torchsaver) [`vllm`](/packages/llm/vllm) [`genai-bench`](/packages/llm/sglang/genai-bench) [`sgl-kernel`](/packages/llm/sglang/sgl-kernel) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/sgl-project/sglang |

| **`sglang:0.4.9`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `sglang` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`flashinfer`](/packages/attention/flash-infer) [`torchao`](/packages/pytorch/torchao) [`cuda-python`](/packages/cuda/cuda-python) [`ffmpeg`](/packages/multimedia/ffmpeg) [`ninja`](/packages/build/ninja) [`torchaudio`](/packages/pytorch/torchaudio) [`causalconv1d`](/packages/ml/mamba/causalconv1d) [`mamba`](/packages/ml/mamba/mamba) [`xgrammar`](/packages/llm/xgrammar) [`flashinfer`](/packages/attention/flash-infer) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`flexprefill`](/packages/attention/flexprefill) [`minference`](/packages/llm/minference) [`torch-memory-saver`](/packages/pytorch/torchsaver) [`vllm`](/packages/llm/vllm) [`genai-bench`](/packages/llm/sglang/genai-bench) [`sgl-kernel`](/packages/llm/sglang/sgl-kernel) |
| &nbsp;&nbsp;&nbsp;Dependants | [`dynamo:0.3.2`](/packages/llm/dynamo/dynamo) [`l4t-dynamo`](/packages/ml/l4t/l4t-dynamo) [`llama-factory`](/packages/llm/llama-factory) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/sgl-project/sglang |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/sglang:0.4.4-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/sglang/tags) | `2025-03-03` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/sglang:0.4.7-r36.4-cu128-24.04`](https://hub.docker.com/r/dustynv/sglang/tags) | `2025-05-06` | `arm64` | `4.9GB` |
| &nbsp;&nbsp;[`dustynv/sglang:r36.4.0`](https://hub.docker.com/r/dustynv/sglang/tags) | `2025-02-07` | `arm64` | `7.3GB` |

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
jetson-containers run $(autotag sglang)

# or explicitly specify one of the container images above
jetson-containers run dustynv/sglang:0.4.7-r36.4-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/sglang:0.4.7-r36.4-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag sglang)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag sglang) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build sglang
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
