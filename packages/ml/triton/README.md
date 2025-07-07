# triton

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`triton:3.5.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The OpenAI `triton` (https://github.com/triton-lang/triton) wheel that's built is saved in the container under `/opt`. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339 |

| **`triton:3.4.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `triton` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) |
| &nbsp;&nbsp;&nbsp;Dependants | [`4k4d:0.0.0`](/packages/3d/gaussian_splatting/4k4d) [`ai-toolkit`](/packages/diffusion/ai-toolkit) [`apex:0.1`](/packages/pytorch/apex) [`audiocraft`](/packages/speech/audiocraft) [`awq:0.1.0`](/packages/llm/awq) [`bitsandbytes:0.39.1`](/packages/llm/bitsandbytes) [`bitsandbytes:0.45.4`](/packages/llm/bitsandbytes) [`bitsandbytes:0.45.5`](/packages/llm/bitsandbytes) [`bitsandbytes:0.46.0`](/packages/llm/bitsandbytes) [`bitsandbytes:0.47.0`](/packages/llm/bitsandbytes) [`bitsandbytes:0.48.0`](/packages/llm/bitsandbytes) [`block_sparse_attn:0.0.1`](/packages/attention/block-sparse-attention) [`cache_dit:0.2.0`](/packages/diffusion/cache_edit) [`causalconv1d:1.6.0`](/packages/ml/mamba/causalconv1d) [`cobra:0.0.1`](/packages/ml/mamba/cobra) [`comfyui`](/packages/diffusion/comfyui) [`cosmos-predict2`](/packages/diffusion/cosmos/cosmos-predict2) [`cosmos-reason1`](/packages/diffusion/cosmos/cosmos-reason1) [`cosmos-transfer1`](/packages/diffusion/cosmos/cosmos-transfer1) [`cosmos1-diffusion-renderer:1.0.4`](/packages/diffusion/cosmos/cosmos_diffusion_renderer) [`deepspeed:0.15.2`](/packages/llm/deepspeed) [`deepspeed:0.18.0`](/packages/llm/deepspeed) [`deepspeed:0.9.5`](/packages/llm/deepspeed) [`dimba:1.0`](/packages/ml/mamba/dimba) [`dynamo:0.3.2`](/packages/llm/dynamo/dynamo) [`easyvolcap:0.0.0`](/packages/3d/gaussian_splatting/easyvolcap) [`exllama:0.1`](/packages/llm/exllama) [`fast_gauss:1.0.0`](/packages/3d/gaussian_splatting/fast_gauss) [`flash-attention:2.5.7`](/packages/attention/flash-attention) [`flash-attention:2.6.3`](/packages/attention/flash-attention) [`flash-attention:2.7.2.post1`](/packages/attention/flash-attention) [`flash-attention:2.7.4.post1`](/packages/attention/flash-attention) [`flash-attention:2.8.0.post2`](/packages/attention/flash-attention) [`flash-attention:2.8.1`](/packages/attention/flash-attention) [`flashinfer:0.2.1.post2`](/packages/attention/flash-infer) [`flashinfer:0.2.2`](/packages/attention/flash-infer) [`flashinfer:0.2.2.post1`](/packages/attention/flash-infer) [`flashinfer:0.2.6.post1`](/packages/attention/flash-infer) [`flashinfer:0.2.7`](/packages/attention/flash-infer) [`flashinfer:0.2.8`](/packages/attention/flash-infer) [`flexprefill:0.1.0`](/packages/attention/flexprefill) [`framepack`](/packages/diffusion/framepack) [`fruitnerf:1.0`](/packages/3d/nerf/fruitnerf) [`genai-bench:0.1.0`](/packages/llm/sglang/genai-bench) [`huggingface_kernels:0.7.0`](/packages/attention/huggingface_kernels) [`hymba`](/packages/llm/hymba) [`isaac-gr00t`](/packages/vla/isaac-gr00t) [`kat:1`](/packages/ml/kans/kat) [`l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) [`l4t-dynamo`](/packages/ml/l4t/l4t-dynamo) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`l4t-text-generation`](/packages/ml/l4t/l4t-text-generation) [`lita`](/packages/vlm/lita) [`llama-factory`](/packages/llm/llama-factory) [`llama-vision`](/packages/vlm/llama-vision) [`llava`](/packages/vlm/llava) [`lobechat`](/packages/llm/lobe_chat) [`local_llm`](/packages/llm/local_llm) [`log-linear-attention:0.0.1`](/packages/attention/log-linear-attention) [`mamba:2.2.5`](/packages/ml/mamba/mamba) [`mambavision:1.0`](/packages/ml/mamba/mambavision) [`minference:0.1.7`](/packages/llm/minference) [`mlc:0.1.0`](/packages/llm/mlc) [`mlc:0.1.1`](/packages/llm/mlc) [`mlc:0.1.2`](/packages/llm/mlc) [`mlc:0.1.3`](/packages/llm/mlc) [`mlc:0.1.4`](/packages/llm/mlc) [`mlc:0.19.0`](/packages/llm/mlc) [`mlc:0.20.0`](/packages/llm/mlc) [`mlc:0.21.0`](/packages/llm/mlc) [`mlstm_kernels:2.0.1`](/packages/ml/xlstm/mlstm_kernels) [`mooncake:0.3.5`](/packages/llm/dynamo/mooncake) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`nerfstudio:1.1.7`](/packages/3d/nerf/nerfstudio) [`nixl:0.3.2`](/packages/llm/dynamo/nixl) [`nvidia_modelopt:0.32.0`](/packages/llm/tensorrt_optimizer/nvidia-modelopt) [`open3d:1.19.0`](/packages/3d/3dvision/open3d) [`openvla`](/packages/vla/openvla) [`openvla:mimicgen`](/packages/vla/openvla) [`paraattention:0.4.0`](/packages/attention/ParaAttention) [`partpacker:0.1.0`](/packages/3d/3dobjects/partpacker) [`plstm:0.1.0`](/packages/ml/xlstm/pltsm) [`prismatic`](/packages/vlm/prismatic) [`pykan:0.2.9`](/packages/ml/kans/pykan) [`pytorch:2.1-all`](/packages/pytorch) [`pytorch:2.2-all`](/packages/pytorch) [`pytorch:2.3-all`](/packages/pytorch) [`pytorch:2.3.1-all`](/packages/pytorch) [`pytorch:2.4-all`](/packages/pytorch) [`pytorch:2.5-all`](/packages/pytorch) [`pytorch:2.6-all`](/packages/pytorch) [`pytorch:2.7-all`](/packages/pytorch) [`pytorch:2.8-all`](/packages/pytorch) [`radial-attention:0.1.0`](/packages/attention/radial-attention) [`robopoint`](/packages/vla/robopoint) [`sage-attention:3.0.0`](/packages/attention/sage-attention) [`sdnext`](/packages/diffusion/sdnext) [`self-forcing`](/packages/diffusion/self-forcing) [`sgl-kernel:0.2.3`](/packages/llm/sglang/sgl-kernel) [`sglang:0.4.4`](/packages/llm/sglang) [`sglang:0.4.6`](/packages/llm/sglang) [`sglang:0.4.9`](/packages/llm/sglang) [`sparc3d:0.1.0`](/packages/3d/3dobjects/sparc3d) [`sparge-attention:0.1.0`](/packages/attention/sparge-attention) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`sudonim:hf`](/packages/llm/sudonim) [`tensorrt_llm:0.12`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.22.0`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`torch-memory-saver:0.0.7`](/packages/pytorch/torchsaver) [`torchao:0.12.0`](/packages/pytorch/torchao) [`torchao:0.13.0`](/packages/pytorch/torchao) [`transformer-engine:2.7`](/packages/ml/transformer-engine) [`videollama:1.0.0`](/packages/vlm/videollama) [`videomambasuite:1.0`](/packages/ml/mamba/videomambasuite) [`vila`](/packages/vlm/vila) [`vllm:0.7.4`](/packages/llm/vllm) [`vllm:0.8.4`](/packages/llm/vllm) [`vllm:0.9.0`](/packages/llm/vllm) [`vllm:0.9.2`](/packages/llm/vllm) [`vllm:0.9.3`](/packages/llm/vllm) [`vllm:v0.8.5.post1`](/packages/llm/vllm) [`voice-pro`](/packages/speech/voice-pro) [`voicecraft`](/packages/speech/voicecraft) [`vscode:torch`](/packages/code/vscode) [`vscode:transformers`](/packages/code/vscode) [`xattention:0.0.1`](/packages/attention/xattention) [`xformers:0.0.32`](/packages/attention/xformers) [`xformers:0.0.33`](/packages/attention/xformers) [`xlstm:2.0.5`](/packages/ml/xlstm/xlstm) [`xtuner`](/packages/vlm/xtuner) [`zigma:1.0`](/packages/ml/mamba/zigma) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The OpenAI `triton` (https://github.com/triton-lang/triton) wheel that's built is saved in the container under `/opt`. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339 |

| **`triton:3.3.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The OpenAI `triton` (https://github.com/triton-lang/triton) wheel that's built is saved in the container under `/opt`. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339 |

| **`triton:3.3.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The OpenAI `triton` (https://github.com/triton-lang/triton) wheel that's built is saved in the container under `/opt`. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339 |

| **`triton:3.2.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The OpenAI `triton` (https://github.com/triton-lang/triton) wheel that's built is saved in the container under `/opt`. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339 |

| **`triton:3.1.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The OpenAI `triton` (https://github.com/triton-lang/triton) wheel that's built is saved in the container under `/opt`. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339 |

| **`triton:3.0.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The OpenAI `triton` (https://github.com/triton-lang/triton) wheel that's built is saved in the container under `/opt`. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339 |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag triton)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host triton:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag triton)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag triton) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build triton
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
