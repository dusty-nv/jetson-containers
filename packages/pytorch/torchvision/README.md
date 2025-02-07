# torchvision

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`torchvision:0.20.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `torchvision` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==36.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.5`](/packages/pytorch) |
| &nbsp;&nbsp;&nbsp;Dependants | [`3d_diffusion_policy`](/packages/robots/3d_diffusion_policy) [`ai-toolkit`](/packages/diffusion/ai-toolkit) [`audiocraft`](/packages/speech/audiocraft) [`auto_awq:0.2.4`](/packages/llm/auto_awq) [`auto_awq:0.2.4-builder`](/packages/llm/auto_awq) [`auto_awq:0.2.6`](/packages/llm/auto_awq) [`auto_awq:0.2.6-builder`](/packages/llm/auto_awq) [`auto_awq:0.2.7.post2`](/packages/llm/auto_awq) [`auto_awq:0.2.7.post2-builder`](/packages/llm/auto_awq) [`auto_awq:0.2.8`](/packages/llm/auto_awq) [`auto_awq:0.2.8-builder`](/packages/llm/auto_awq) [`auto_gptq:0.7.1`](/packages/llm/auto_gptq) [`auto_gptq:0.7.1-builder`](/packages/llm/auto_gptq) [`auto_gptq:0.8.0`](/packages/llm/auto_gptq) [`auto_gptq:0.8.0-builder`](/packages/llm/auto_gptq) [`awq:0.1.0`](/packages/llm/awq) [`awq:0.1.0-builder`](/packages/llm/awq) [`bitsandbytes:0.39.1`](/packages/llm/bitsandbytes) [`bitsandbytes:0.39.1-builder`](/packages/llm/bitsandbytes) [`bitsandbytes:0.44.1`](/packages/llm/bitsandbytes) [`bitsandbytes:0.44.1-builder`](/packages/llm/bitsandbytes) [`bitsandbytes:0.45.1`](/packages/llm/bitsandbytes) [`bitsandbytes:0.45.1-builder`](/packages/llm/bitsandbytes) [`causalconv1d:1.4.0`](/packages/ml/mamba/causalconv1d) [`causalconv1d:1.4.0-builder`](/packages/ml/mamba/causalconv1d) [`causalconv1d:1.6.0`](/packages/ml/mamba/causalconv1d) [`causalconv1d:1.6.0-builder`](/packages/ml/mamba/causalconv1d) [`clip_trt`](/packages/vit/clip_trt) [`cobra:0.0.1`](/packages/ml/mamba/cobra) [`comfyui`](/packages/diffusion/comfyui) [`cosmos`](/packages/diffusion/cosmos) [`crossformer`](/packages/robots/crossformer) [`deepspeed:0.15.2`](/packages/llm/deepspeed) [`deepspeed:0.15.2-builder`](/packages/llm/deepspeed) [`deepspeed:0.16.3`](/packages/llm/deepspeed) [`deepspeed:0.16.3-builder`](/packages/llm/deepspeed) [`deepspeed:0.9.5`](/packages/llm/deepspeed) [`deepspeed:0.9.5-builder`](/packages/llm/deepspeed) [`diffusers:0.30.2`](/packages/diffusion/diffusers) [`diffusers:0.30.2-builder`](/packages/diffusion/diffusers) [`diffusers:0.31.0`](/packages/diffusion/diffusers) [`diffusers:0.31.0-builder`](/packages/diffusion/diffusers) [`diffusers:0.32.3`](/packages/diffusion/diffusers) [`diffusers:0.32.3-builder`](/packages/diffusion/diffusers) [`diffusion_policy`](/packages/robots/diffusion_policy) [`dimba:1.0`](/packages/ml/mamba/dimba) [`efficientvit`](/packages/vit/efficientvit) [`fruitnerf:1.0`](/packages/nerf/fruitnerf) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`gsplat:1.3.0`](/packages/nerf/gsplat) [`gsplat:1.3.0-builder`](/packages/nerf/gsplat) [`gsplat:1.5.0`](/packages/nerf/gsplat) [`gsplat:1.5.0-builder`](/packages/nerf/gsplat) [`hloc:1.4`](/packages/nerf/hloc) [`hloc:1.4-builder`](/packages/nerf/hloc) [`hloc:1.5`](/packages/nerf/hloc) [`hloc:1.5-builder`](/packages/nerf/hloc) [`hymba`](/packages/llm/hymba) [`jetson-copilot`](/packages/rag/jetson-copilot) [`jetson-inference:foxy`](/packages/ml/jetson-inference) [`jetson-inference:galactic`](/packages/ml/jetson-inference) [`jetson-inference:humble`](/packages/ml/jetson-inference) [`jetson-inference:iron`](/packages/ml/jetson-inference) [`jetson-inference:jazzy`](/packages/ml/jetson-inference) [`jetson-inference:main`](/packages/ml/jetson-inference) [`kat:1`](/packages/vit/kat) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`lerobot`](/packages/robots/lerobot) [`libcom:0.1.1`](/packages/multimedia/libcom) [`lita`](/packages/vlm/lita) [`llama-factory`](/packages/llm/llama-factory) [`llama-vision`](/packages/vlm/llama-vision) [`llava`](/packages/vlm/llava) [`local_llm`](/packages/llm/local_llm) [`mamba:2.2.2`](/packages/ml/mamba/mamba) [`mamba:2.2.2-builder`](/packages/ml/mamba/mamba) [`mamba:2.2.5`](/packages/ml/mamba/mamba) [`mamba:2.2.5-builder`](/packages/ml/mamba/mamba) [`mambavision:1.0`](/packages/ml/mamba/mambavision) [`mimicgen`](/packages/robots/mimicgen) [`minigpt4`](/packages/vlm/minigpt4) [`mlc:0.1.0`](/packages/llm/mlc) [`mlc:0.1.0-builder`](/packages/llm/mlc) [`mlc:0.1.1`](/packages/llm/mlc) [`mlc:0.1.1-builder`](/packages/llm/mlc) [`mlc:0.1.2`](/packages/llm/mlc) [`mlc:0.1.2-builder`](/packages/llm/mlc) [`mlc:0.1.3`](/packages/llm/mlc) [`mlc:0.1.3-builder`](/packages/llm/mlc) [`mlc:0.1.4`](/packages/llm/mlc) [`mlc:0.1.4-builder`](/packages/llm/mlc) [`mlc:0.19.0`](/packages/llm/mlc) [`mlc:0.19.0-builder`](/packages/llm/mlc) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`nemo:1.23.0`](/packages/llm/nemo) [`nemo:2.0.0`](/packages/llm/nemo) [`nemo:2.0.0-builder`](/packages/llm/nemo) [`nerfacc:0.5.3`](/packages/nerf/nerfacc) [`nerfacc:0.5.3-builder`](/packages/nerf/nerfacc) [`nerfacc:0.5.4`](/packages/nerf/nerfacc) [`nerfacc:0.5.4-builder`](/packages/nerf/nerfacc) [`nerfstudio:1.1.4`](/packages/nerf/nerfstudio) [`nerfstudio:1.1.4-builder`](/packages/nerf/nerfstudio) [`nerfstudio:1.1.5`](/packages/nerf/nerfstudio) [`nerfstudio:1.1.5-builder`](/packages/nerf/nerfstudio) [`nerfstudio:1.1.6`](/packages/nerf/nerfstudio) [`nerfstudio:1.1.6-builder`](/packages/nerf/nerfstudio) [`octo`](/packages/robots/octo) [`onnxruntime_genai:0.6.0`](/packages/ml/onnxruntime_genai) [`onnxruntime_genai:0.6.0-builder`](/packages/ml/onnxruntime_genai) [`openvla`](/packages/robots/openvla) [`openvla:mimicgen`](/packages/robots/openvla) [`optimum`](/packages/llm/optimum) [`pixsfm:1.0`](/packages/nerf/pixsfm) [`pixsfm:1.0-builder`](/packages/nerf/pixsfm) [`prismatic`](/packages/vlm/prismatic) [`pycolmap:3.10`](/packages/nerf/pycolmap) [`pycolmap:3.10-builder`](/packages/nerf/pycolmap) [`pycolmap:3.11.1`](/packages/nerf/pycolmap) [`pycolmap:3.11.1-builder`](/packages/nerf/pycolmap) [`pycolmap:3.12`](/packages/nerf/pycolmap) [`pycolmap:3.12-builder`](/packages/nerf/pycolmap) [`pycolmap:3.8`](/packages/nerf/pycolmap) [`pycolmap:3.8-builder`](/packages/nerf/pycolmap) [`robomimic`](/packages/robots/robomimic) [`sam`](/packages/vit/sam) [`sapiens`](/packages/vit/sapiens) [`sglang`](/packages/llm/sglang) [`shape-llm`](/packages/vlm/shape-llm) [`stable-diffusion`](/packages/diffusion/stable-diffusion) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`tensorrt_llm:0.12`](/packages/llm/tensorrt_llm) [`tensorrt_llm:0.12-builder`](/packages/llm/tensorrt_llm) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`tinycudann:1.7`](/packages/nerf/tinycudann) [`tinycudann:1.7-builder`](/packages/nerf/tinycudann) [`torch2trt`](/packages/pytorch/torch2trt) [`torch_tensorrt`](/packages/pytorch/torch_tensorrt) [`transformer-engine:1.13`](/packages/ml/transformer-engine) [`transformer-engine:1.13-builder`](/packages/ml/transformer-engine) [`transformer-engine:2.0`](/packages/ml/transformer-engine) [`transformer-engine:2.0-builder`](/packages/ml/transformer-engine) [`transformers:4.48.2`](/packages/llm/transformers) [`transformers:git`](/packages/llm/transformers) [`transformers:nvgpt`](/packages/llm/transformers) [`vhacdx:0.0.8.post1`](/packages/nerf/vhacdx) [`vhacdx:0.0.8.post1-builder`](/packages/nerf/vhacdx) [`vhacdx:0.0.9.post1`](/packages/nerf/vhacdx) [`vhacdx:0.0.9.post1-builder`](/packages/nerf/vhacdx) [`videomambasuite:1.0`](/packages/ml/mamba/videomambasuite) [`vila`](/packages/vlm/vila) [`vllm:0.7.2`](/packages/llm/vllm) [`vllm:0.7.2-builder`](/packages/llm/vllm) [`voicecraft`](/packages/speech/voicecraft) [`whisper_trt`](/packages/speech/whisper_trt) [`whisperx`](/packages/speech/whisperx) [`xtts`](/packages/speech/xtts) [`xtuner`](/packages/vlm/xtuner) [`zigma:1.0`](/packages/ml/mamba/zigma) [`zigma:1.0-builder`](/packages/ml/mamba/zigma) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/torchvision:r32.7.1`](https://hub.docker.com/r/dustynv/torchvision/tags) | `2023-12-14` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/torchvision:r35.2.1`](https://hub.docker.com/r/dustynv/torchvision/tags) | `2023-12-11` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/torchvision:r35.3.1`](https://hub.docker.com/r/dustynv/torchvision/tags) | `2023-12-14` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/torchvision:r35.4.1`](https://hub.docker.com/r/dustynv/torchvision/tags) | `2023-11-05` | `arm64` | `5.4GB` |

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
jetson-containers run $(autotag torchvision)

# or explicitly specify one of the container images above
jetson-containers run dustynv/torchvision:r35.3.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/torchvision:r35.3.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag torchvision)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag torchvision) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build torchvision
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
