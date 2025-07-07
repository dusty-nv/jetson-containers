# huggingface_hub

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`huggingface_hub`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dependants | [`3d_diffusion_policy`](/packages/diffusion/3d_diffusion_policy) [`ai-toolkit`](/packages/diffusion/ai-toolkit) [`apex:0.1`](/packages/pytorch/apex) [`audiocraft`](/packages/speech/audiocraft) [`awq:0.1.0`](/packages/llm/awq) [`bitsandbytes:0.39.1`](/packages/llm/bitsandbytes) [`bitsandbytes:0.45.4`](/packages/llm/bitsandbytes) [`bitsandbytes:0.45.5`](/packages/llm/bitsandbytes) [`bitsandbytes:0.46.0`](/packages/llm/bitsandbytes) [`bitsandbytes:0.47.0`](/packages/llm/bitsandbytes) [`bitsandbytes:0.48.0`](/packages/llm/bitsandbytes) [`block_sparse_attn:0.0.1`](/packages/attention/block-sparse-attention) [`cache_dit:0.2.0`](/packages/diffusion/cache_edit) [`chatterbox-tts`](/packages/speech/chatterbox-tts) [`clip_trt`](/packages/vit/clip_trt) [`cobra:0.0.1`](/packages/ml/mamba/cobra) [`comfyui`](/packages/diffusion/comfyui) [`cosmos-predict2`](/packages/diffusion/cosmos/cosmos-predict2) [`cosmos-reason1`](/packages/diffusion/cosmos/cosmos-reason1) [`cosmos-transfer1`](/packages/diffusion/cosmos/cosmos-transfer1) [`cosmos1-diffusion-renderer:1.0.4`](/packages/diffusion/cosmos/cosmos_diffusion_renderer) [`crossformer`](/packages/vla/crossformer) [`deepspeed:0.15.2`](/packages/llm/deepspeed) [`deepspeed:0.18.0`](/packages/llm/deepspeed) [`deepspeed:0.9.5`](/packages/llm/deepspeed) [`diffusers:0.35.0`](/packages/diffusion/diffusers) [`diffusion_policy`](/packages/diffusion/diffusion_policy) [`dimba:1.0`](/packages/ml/mamba/dimba) [`dynamo:0.3.2`](/packages/llm/dynamo/dynamo) [`efficientvit`](/packages/vit/efficientvit) [`exllama:0.1`](/packages/llm/exllama) [`faster-whisper`](/packages/speech/faster-whisper) [`flash-attention:2.5.7`](/packages/attention/flash-attention) [`flash-attention:2.6.3`](/packages/attention/flash-attention) [`flash-attention:2.7.2.post1`](/packages/attention/flash-attention) [`flash-attention:2.7.4.post1`](/packages/attention/flash-attention) [`flash-attention:2.8.0.post2`](/packages/attention/flash-attention) [`flash-attention:2.8.1`](/packages/attention/flash-attention) [`flexprefill:0.1.0`](/packages/attention/flexprefill) [`framepack`](/packages/diffusion/framepack) [`fruitnerf:1.0`](/packages/3d/nerf/fruitnerf) [`genai-bench:0.1.0`](/packages/llm/sglang/genai-bench) [`gptqmodel:3.0.1`](/packages/llm/gptqmodel) [`huggingface_kernels:0.7.0`](/packages/attention/huggingface_kernels) [`hymba`](/packages/llm/hymba) [`isaac-gr00t`](/packages/vla/isaac-gr00t) [`isaaclab:2.2.0`](/packages/sim/isaac-sim/isaac-lab) [`isaacsim:5.0.0`](/packages/sim/isaac-sim) [`jetson-copilot`](/packages/rag/jetson-copilot) [`kat:1`](/packages/ml/kans/kat) [`kokoro-tts:fastapi`](/packages/speech/kokoro-tts/kokoro-tts-fastapi) [`kokoro-tts:hf`](/packages/speech/kokoro-tts/kokoro-tts-hf) [`ktransformers:0.3.3`](/packages/llm/ktransformers) [`l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) [`l4t-dynamo`](/packages/ml/l4t/l4t-dynamo) [`l4t-text-generation`](/packages/ml/l4t/l4t-text-generation) [`langchain`](/packages/rag/langchain) [`langchain:samples`](/packages/rag/langchain) [`lerobot`](/packages/robots/lerobot) [`lita`](/packages/vlm/lita) [`llama-factory`](/packages/llm/llama-factory) [`llama-vision`](/packages/vlm/llama-vision) [`llama_cpp:0.2.57`](/packages/llm/llama_cpp) [`llama_cpp:0.2.70`](/packages/llm/llama_cpp) [`llama_cpp:0.2.83`](/packages/llm/llama_cpp) [`llama_cpp:0.2.90`](/packages/llm/llama_cpp) [`llama_cpp:0.3.1`](/packages/llm/llama_cpp) [`llama_cpp:0.3.2`](/packages/llm/llama_cpp) [`llama_cpp:0.3.5`](/packages/llm/llama_cpp) [`llama_cpp:0.3.6`](/packages/llm/llama_cpp) [`llama_cpp:0.3.7`](/packages/llm/llama_cpp) [`llama_cpp:0.3.8`](/packages/llm/llama_cpp) [`llama_cpp:0.3.9`](/packages/llm/llama_cpp) [`llama_cpp:0.4.0`](/packages/llm/llama_cpp) [`llama_cpp:b5255`](/packages/llm/llama_cpp) [`llama_cpp:b5833`](/packages/llm/llama_cpp) [`llava`](/packages/vlm/llava) [`lobechat`](/packages/llm/lobe_chat) [`local_llm`](/packages/llm/local_llm) [`log-linear-attention:0.0.1`](/packages/attention/log-linear-attention) [`mamba:2.2.5`](/packages/ml/mamba/mamba) [`mambavision:1.0`](/packages/ml/mamba/mambavision) [`minference:0.1.7`](/packages/llm/minference) [`minigpt4`](/packages/vlm/minigpt4) [`mlc:0.1.0`](/packages/llm/mlc) [`mlc:0.1.1`](/packages/llm/mlc) [`mlc:0.1.2`](/packages/llm/mlc) [`mlc:0.1.3`](/packages/llm/mlc) [`mlc:0.1.4`](/packages/llm/mlc) [`mlc:0.19.0`](/packages/llm/mlc) [`mlc:0.20.0`](/packages/llm/mlc) [`mlc:0.21.0`](/packages/llm/mlc) [`mlstm_kernels:2.0.1`](/packages/ml/xlstm/mlstm_kernels) [`mooncake:0.3.5`](/packages/llm/dynamo/mooncake) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`nemo`](/packages/llm/nemo) [`nerfstudio:1.1.7`](/packages/3d/nerf/nerfstudio) [`nixl:0.3.2`](/packages/llm/dynamo/nixl) [`nvdiffrast:0.3.4`](/packages/diffusion/cosmos/cosmos_diffusion_renderer/nvdiffrast) [`nvidia_modelopt:0.32.0`](/packages/llm/tensorrt_optimizer/nvidia-modelopt) [`octo`](/packages/vla/octo) [`onnxruntime_genai:0.8.5`](/packages/ml/onnxruntime_genai) [`openpi`](/packages/robots/openpi) [`openvla`](/packages/vla/openvla) [`openvla:mimicgen`](/packages/vla/openvla) [`optimum`](/packages/llm/optimum) [`paraattention:0.4.0`](/packages/attention/ParaAttention) [`partpacker:0.1.0`](/packages/3d/3dobjects/partpacker) [`plstm:0.1.0`](/packages/ml/xlstm/pltsm) [`prismatic`](/packages/vlm/prismatic) [`pytorch:2.1-all`](/packages/pytorch) [`pytorch:2.2-all`](/packages/pytorch) [`pytorch:2.3-all`](/packages/pytorch) [`pytorch:2.3.1-all`](/packages/pytorch) [`pytorch:2.4-all`](/packages/pytorch) [`pytorch:2.5-all`](/packages/pytorch) [`pytorch:2.6-all`](/packages/pytorch) [`pytorch:2.7-all`](/packages/pytorch) [`pytorch:2.8-all`](/packages/pytorch) [`radial-attention:0.1.0`](/packages/attention/radial-attention) [`robopoint`](/packages/vla/robopoint) [`sage-attention:3.0.0`](/packages/attention/sage-attention) [`sdnext`](/packages/diffusion/sdnext) [`self-forcing`](/packages/diffusion/self-forcing) [`sgl-kernel:0.2.3`](/packages/llm/sglang/sgl-kernel) [`sglang:0.4.4`](/packages/llm/sglang) [`sglang:0.4.6`](/packages/llm/sglang) [`sglang:0.4.9`](/packages/llm/sglang) [`sparc3d:0.1.0`](/packages/3d/3dobjects/sparc3d) [`sparge-attention:0.1.0`](/packages/attention/sparge-attention) [`spark-tts`](/packages/speech/spark-tts) [`speaches`](/packages/speech/speaches) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`sudonim:hf`](/packages/llm/sudonim) [`tensorrt_llm:0.12`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.22.0`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`transformer-engine:2.7`](/packages/ml/transformer-engine) [`transformers:4.53.1`](/packages/llm/transformers) [`videollama:1.0.0`](/packages/vlm/videollama) [`videomambasuite:1.0`](/packages/ml/mamba/videomambasuite) [`vila`](/packages/vlm/vila) [`vllm:0.7.4`](/packages/llm/vllm) [`vllm:0.8.4`](/packages/llm/vllm) [`vllm:0.9.0`](/packages/llm/vllm) [`vllm:0.9.2`](/packages/llm/vllm) [`vllm:0.9.3`](/packages/llm/vllm) [`vllm:v0.8.5.post1`](/packages/llm/vllm) [`voice-pro`](/packages/speech/voice-pro) [`voicecraft`](/packages/speech/voicecraft) [`vscode:transformers`](/packages/code/vscode) [`whisperx`](/packages/speech/whisperx) [`wyoming-whisper:2.5.0`](/packages/smart-home/wyoming/wyoming-whisper) [`wyoming-whisper:master`](/packages/smart-home/wyoming/wyoming-whisper) [`xattention:0.0.1`](/packages/attention/xattention) [`xformers:0.0.32`](/packages/attention/xformers) [`xformers:0.0.33`](/packages/attention/xformers) [`xgrammar:0.1.15`](/packages/llm/xgrammar) [`xgrammar:0.1.18`](/packages/llm/xgrammar) [`xgrammar:0.1.19`](/packages/llm/xgrammar) [`xgrammar:0.1.20`](/packages/llm/xgrammar) [`xgrammar:0.1.21`](/packages/llm/xgrammar) [`xlstm:2.0.5`](/packages/ml/xlstm/xlstm) [`xtts`](/packages/speech/xtts) [`xtuner`](/packages/vlm/xtuner) [`zigma:1.0`](/packages/ml/mamba/zigma) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/huggingface_hub:r32.7.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) `(2023-12-15, 0.4GB)`<br>[`dustynv/huggingface_hub:r35.2.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) `(2023-12-15, 5.0GB)`<br>[`dustynv/huggingface_hub:r35.3.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) `(2023-09-07, 5.0GB)`<br>[`dustynv/huggingface_hub:r35.4.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) `(2023-10-07, 5.0GB)` |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/huggingface/huggingface_hub |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/huggingface_hub:r32.7.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) | `2023-12-15` | `arm64` | `0.4GB` |
| &nbsp;&nbsp;[`dustynv/huggingface_hub:r35.2.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) | `2023-12-15` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/huggingface_hub:r35.3.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) | `2023-09-07` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/huggingface_hub:r35.4.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) | `2023-10-07` | `arm64` | `5.0GB` |

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
jetson-containers run $(autotag huggingface_hub)

# or explicitly specify one of the container images above
jetson-containers run dustynv/huggingface_hub:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/huggingface_hub:r35.2.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag huggingface_hub)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag huggingface_hub) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build huggingface_hub
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
