# torchao

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`torchao:0.12.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `torchao` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) [`triton`](/packages/ml/triton) |
| &nbsp;&nbsp;&nbsp;Dependants | [`4k4d:0.0.0`](/packages/3d/gaussian_splatting/4k4d) [`block_sparse_attn:0.0.1`](/packages/attention/block-sparse-attention) [`comfyui`](/packages/diffusion/comfyui) [`cosmos-reason1`](/packages/diffusion/cosmos/cosmos-reason1) [`dynamo:0.3.2`](/packages/llm/dynamo/dynamo) [`easyvolcap:0.0.0`](/packages/3d/gaussian_splatting/easyvolcap) [`fast_gauss:1.0.0`](/packages/3d/gaussian_splatting/fast_gauss) [`flexprefill:0.1.0`](/packages/attention/flexprefill) [`framepack`](/packages/diffusion/framepack) [`genai-bench:0.1.0`](/packages/llm/sglang/genai-bench) [`l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) [`l4t-dynamo`](/packages/ml/l4t/l4t-dynamo) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`llama-factory`](/packages/llm/llama-factory) [`lobechat`](/packages/llm/lobe_chat) [`minference:0.1.7`](/packages/llm/minference) [`mooncake:0.3.5`](/packages/llm/dynamo/mooncake) [`nixl:0.3.2`](/packages/llm/dynamo/nixl) [`nvidia_modelopt:0.32.0`](/packages/llm/tensorrt_optimizer/nvidia-modelopt) [`open3d:1.19.0`](/packages/3d/3dvision/open3d) [`paraattention:0.4.0`](/packages/attention/ParaAttention) [`pytorch:2.1-all`](/packages/pytorch) [`pytorch:2.2-all`](/packages/pytorch) [`pytorch:2.3-all`](/packages/pytorch) [`pytorch:2.3.1-all`](/packages/pytorch) [`pytorch:2.4-all`](/packages/pytorch) [`pytorch:2.5-all`](/packages/pytorch) [`pytorch:2.6-all`](/packages/pytorch) [`pytorch:2.7-all`](/packages/pytorch) [`pytorch:2.8-all`](/packages/pytorch) [`sage-attention:3.0.0`](/packages/attention/sage-attention) [`sdnext`](/packages/diffusion/sdnext) [`self-forcing`](/packages/diffusion/self-forcing) [`sgl-kernel:0.2.3`](/packages/llm/sglang/sgl-kernel) [`sglang:0.4.4`](/packages/llm/sglang) [`sglang:0.4.6`](/packages/llm/sglang) [`sglang:0.4.9`](/packages/llm/sglang) [`sparge-attention:0.1.0`](/packages/attention/sparge-attention) [`sudonim:hf`](/packages/llm/sudonim) [`tensorrt_llm:0.12`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.22.0`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`videollama:1.0.0`](/packages/vlm/videollama) [`vllm:0.7.4`](/packages/llm/vllm) [`vllm:0.8.4`](/packages/llm/vllm) [`vllm:0.9.0`](/packages/llm/vllm) [`vllm:0.9.2`](/packages/llm/vllm) [`vllm:0.9.3`](/packages/llm/vllm) [`vllm:v0.8.5.post1`](/packages/llm/vllm) [`voice-pro`](/packages/speech/voice-pro) [`vscode:torch`](/packages/code/vscode) [`xattention:0.0.1`](/packages/attention/xattention) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`torchao:0.13.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) [`triton`](/packages/ml/triton) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/torchao:0.10.0-amd64-cu128-24.04`](https://hub.docker.com/r/dustynv/torchao/tags) | `2025-04-02` | `amd64` | `6.9GB` |
| &nbsp;&nbsp;[`dustynv/torchao:0.11.0-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/torchao/tags) | `2025-02-26` | `arm64` | `3.5GB` |

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
jetson-containers run $(autotag torchao)

# or explicitly specify one of the container images above
jetson-containers run dustynv/torchao:0.10.0-amd64-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/torchao:0.10.0-amd64-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag torchao)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag torchao) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build torchao
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
