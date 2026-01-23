[![a header for a software project about building containers for AI and machine learning](https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/header_blueprint_rainbow.jpg)](https://www.jetson-ai-lab.com)

[![jetson-ai-lab.io status](https://img.shields.io/website?label=jetson-ai-lab.io&url=https%3A%2F%2Fpypi.jetson-ai-lab.io&up_message=up&up_color=brightgreen&down_message=down&down_color=red)](https://pypi.jetson-ai-lab.io)
# CUDA Containers for Edge AI & Robotics

Modular container build system that provides the latest [**AI/ML packages**](https://pypi.jetson-ai-lab.io/) for [NVIDIA Jetson](https://jetson-ai-lab.com) :rocket::robot:

|                    |                                                                                |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **ML**             | [`pytorch`](packages/pytorch) [`tensorflow`](packages/ml/tensorflow) [`jax`](packages/ml/jax) [`onnxruntime`](packages/ml/onnxruntime) [`deepstream`](packages/cv/deepstream) [`holoscan`](packages/cv/holoscan) [`CTranslate2`](packages/ml/ctranslate2) [`JupyterLab`](packages/ml/jupyterlab)                                                                                                                                                                                                                                                                                                                                                                                                       |
| **LLM**            | [`SGLang`](packages/llm/sglang) [`vLLM`](packages/llm/vllm) [`MLC`](packages/llm/mlc) [`AWQ`](packages/llm/awq) [`transformers`](packages/llm/transformers) [`text-generation-webui`](packages/llm/text-generation-webui) [`ollama`](packages/llm/ollama) [`llama.cpp`](packages/llm/llama_cpp) [`llama-factory`](packages/llm/llama-factory) [`exllama`](packages/llm/exllama) [`AutoGPTQ`](packages/llm/auto_gptq) [`FlashAttention`](packages/attention/flash-attention) [`DeepSpeed`](packages/llm/deepspeed) [`bitsandbytes`](packages/llm/bitsandbytes) [`xformers`](packages/llm/xformers)                                                                                                      |
| **VLM**            | [`llava`](packages/vlm/llava) [`llama-vision`](packages/vlm/llama-vision) [`VILA`](packages/vlm/vila) [`LITA`](packages/vlm/lita) [`NanoLLM`](packages/llm/nano_llm) [`ShapeLLM`](packages/vlm/shape-llm) [`Prismatic`](packages/vlm/prismatic) [`xtuner`](packages/vlm/xtuner) [`gemma_vlm`](packages/vlm/gemma_vlm)                                                                                                                                                                                                                                                                                                                                                                                  |
| **VIT**            | [`NanoOWL`](packages/vit/nanoowl) [`NanoSAM`](packages/vit/nanosam) [`Segment Anything (SAM)`](packages/vit/sam) [`Track Anything (TAM)`](packages/vit/tam) [`clip_trt`](packages/vit/clip_trt)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **RAG**            | [`llama-index`](packages/rag/llama-index) [`langchain`](packages/rag/langchain) [`jetson-copilot`](packages/rag/jetson-copilot) [`NanoDB`](packages/vectordb/nanodb) [`FAISS`](packages/vectordb/faiss) [`RAFT`](packages/ml/rapids/raft)                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **L4T**            | [`l4t-pytorch`](packages/ml/l4t/l4t-pytorch) [`l4t-tensorflow`](packages/ml/l4t/l4t-tensorflow) [`l4t-ml`](packages/ml/l4t/l4t-ml) [`l4t-diffusion`](packages/ml/l4t/l4t-diffusion) [`l4t-text-generation`](packages/ml/l4t/l4t-text-generation)                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **CUDA**           | [`cupy`](packages/numeric/cupy) [`cuda-python`](packages/cuda/cuda-python) [`pycuda`](packages/cuda/pycuda) [`cv-cuda`](packages/cv/cv-cuda) [`opencv:cuda`](packages/cv/opencv) [`numba`](packages/numeric/numba)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Robotics**       | [`ROS`](packages/robots/ros) [`LeRobot`](packages/robots/lerobot) [`OpenVLA`](packages/vla/openvla) [`3D Diffusion Policy`](packages/diffusion/3d_diffusion_policy) [`Crossformer`](packages/diffusion/crossformer) [`MimicGen`](packages/sim/mimicgen) [`OpenDroneMap`](packages/robots/opendronemap) [`ZED`](packages/hardware/zed)  [`openpi`](packages/robots/openpi)                                                                                                                                                                                                                                                                                                                              |
| **Simulation**     | [`Isaac Sim`](packages/sim/isaac-sim) [`Genesis`](packages/sim/genesis) [`Habitat Sim`](packages/sim/habitat-sim) [`MimicGen`](packages/sim/mimicgen) [`MuJoCo`](packages/sim/mujoco) [`PhysX`](packages/sim/physx) [`Protomotions`](packages/sim/protomotions) [`RoboGen`](packages/sim/robogen) [`RoboMimic`](packages/sim/robomimic) [`RoboSuite`](packages/sim/robosuite) [`Sapien`](packages/sim/sapien)                                                                                                                                                                                                                                                                                          |
| **Graphics**       | [`3D Diffusion Policy`](packages/diffusion/3d_diffusion_policy) [`AI Toolkit`](packages/diffusion/ai-toolkit) [`ComfyUI`](packages/diffusion/comfyui) [`Cosmos`](packages/diffusion/cosmos) [`Diffusers`](packages/diffusion/diffusers) [`Diffusion Policy`](packages/diffusion/diffusion_policy) [`FramePack`](packages/diffusion/framepack) [`Small Stable Diffusion`](packages/diffusion/small-stable-diffusion) [`Stable Diffusion`](packages/diffusion/stable-diffusion) [`Stable Diffusion WebUI`](packages/diffusion/stable-diffusion-webui) [`SD.Next`](packages/diffusion/sdnext) [`nerfstudio`](packages/nerf/nerfstudio) [`meshlab`](packages/nerf/meshlab) [`gsplat`](packages/nerf/gsplat) |
| **Mamba**          | [`mamba`](packages/ml/mamba) [`mambavision`](packages/ml/mamba/mambavision) [`cobra`](packages/ml/mamba/cobra) [`dimba`](packages/ml/mamba/dimba) [`videomambasuite`](packages/ml/mamba/videomambasuite)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **KANs**           | [`pykan`](packages/ml/kans/pykan) [`kat`](packages/ml/kans/kat)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **xLTSM**          | [`xltsm`](packages/ml/xltsm/xltsm) [`mlstm_kernels`](packages/ml/xltsm/mlstm_kernels)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Speech**         | [`whisper`](packages/speech/whisper) [`whisper_trt`](packages/speech/whisper_trt) [`piper`](packages/speech/piper-tts) [`riva`](packages/speech/riva-client) [`audiocraft`](packages/speech/audiocraft) [`voicecraft`](packages/speech/voicecraft) [`xtts`](packages/speech/xtts)                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Home/IoT**       | [`homeassistant-core`](packages/smart-home/homeassistant-core) [`wyoming-whisper`](packages/smart-home/wyoming/wyoming-whisper) [`wyoming-openwakeword`](packages/smart-home/wyoming/openwakeword) [`wyoming-piper`](packages/smart-home/wyoming/piper)                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **3DPrintObjects** | [`PartPacker`](packages/objects/partpacker) [`Sparc3D`](packages/objects/sparc3d)                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

See the [**`packages`**](packages) directory for the full list, including pre-built container images for JetPack/L4T.

Using the included tools, you can easily combine packages together for building your own containers.  Want to run ROS2 with PyTorch and Transformers?  No problem - just do the [system setup](/docs/setup.md), and build it on your Jetson:

```bash
$ jetson-containers build --name=my_container pytorch transformers ros:humble-desktop
```

There are shortcuts for running containers too - this will pull or build a [`l4t-pytorch`](packages/l4t/l4t-pytorch) image that's compatible:

```bash
$ jetson-containers run $(autotag l4t-pytorch)
```
> <sup>[`jetson-containers run`](/docs/run.md) launches [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some added defaults (like `--runtime nvidia`, mounted `/data` cache and devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

If you look at any package's readme (like [`l4t-pytorch`](packages/l4t/l4t-pytorch)), it will have detailed instructions for running it.

#### Changing CUDA Versions

You can rebuild the container stack for different versions of CUDA by setting the `CUDA_VERSION` variable:

```bash
CUDA_VERSION=12.6 jetson-containers build transformers
```

It will then go off and either pull or build all the dependencies needed, including PyTorch and other packages that would be time-consuming to compile.  There is a [Pip server](/docs/build.md#pip-server) that caches the wheels to accelerate builds.  You can also request specific versions of cuDNN, TensorRT, Python, and PyTorch with similar environment variables like [here](/docs/build.md#changing-versions).

## Documentation

<a href="https://www.jetson-ai-lab.com"><img align="right" width="200" height="200" src="https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/images/JON_Gen-AI-panels.png"></a>

* [Package List](/packages)
* [Package Definitions](/docs/packages.md)
* [System Setup](/docs/setup.md)
* [Building Containers](/docs/build.md)
* [Running Containers](/docs/run.md)

Check out the tutorials at the [**Jetson Generative AI Lab**](https://www.jetson-ai-lab.com)!

## Getting Started

Refer to the [System Setup](/docs/setup.md) page for tips about setting up your Docker daemon and memory/storage tuning.

```bash
# install the container tools
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

# automatically pull & run any container
jetson-containers run $(autotag l4t-pytorch)
```

Or you can manually run a [container image](https://hub.docker.com/r/dustynv) of your choice without using the helper scripts above:

```bash
sudo docker run --runtime nvidia -it --rm --network=host dustynv/l4t-pytorch:r36.2.0
```

Looking for the old jetson-containers?   See the [`legacy`](https://github.com/dusty-nv/jetson-containers/tree/legacy) branch.


### Only Tested and supported Jetpack 6.2 (Cuda 12.6) and JetPack 7 (CUDA 13.x).

> [!NOTE]
> Ubuntu 24.04 containers for JetPack 6 and JetPack 7 are now available (with CUDA support)
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`LSB_RELEASE=24.04 jetson-containers build pytorch:2.8`
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`jetson-containers run dustynv/pytorch:2.8-r36.4-cu128-24.04`
>
> ARM SBSA (Server Base System Architecture) is supported for GH200 / GB200.
> To install CUDA 13.0 SBSA wheels for Python 3.12 / 24.04:
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`uv pip install torch torchvision torchaudio \`
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`--index-url https://pypi.jetson-ai-lab.io/sbsa/cu129`
>
> See the **[`Ubuntu 24.04`](/docs/build.md#2404-containers)** section of the docs for details and a list of available containers 🤗
> Thanks to all our contributors from **[`Discord`](https://discord.gg/BmqNSK4886)** and AI community for their support 🤗


## Code Style

The project uses automated code formatting tools to maintain consistent code style. See [Code Style Guide](docs/code-style.md) for details on:
- Setting up formatting tools
- Adding your package to formatting checks
- Troubleshooting common issues


## Gallery

<a href="https://www.youtube.com/watch?v=UOjqF3YCGkY"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_llava_clip.gif"></a>
> [Multimodal Voice Chat with LLaVA-1.5 13B on NVIDIA Jetson AGX Orin](https://www.youtube.com/watch?v=9ObzbbBTbcc) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

<br/>

<a href="https://www.youtube.com/watch?v=hswNSZTvEFE"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_70b_yt.jpg" width="800px"></a>
> [Interactive Voice Chat with Llama-2-70B on NVIDIA Jetson AGX Orin](https://www.youtube.com/watch?v=wzLHAgDxMjQ) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

<br/>

<a href="https://www.youtube.com/watch?v=OJT-Ax0CkhU"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/nanodb_tennis.jpg"></a>
> [Realtime Multimodal VectorDB on NVIDIA Jetson](https://www.youtube.com/watch?v=ayqKpQNd1Jw) (container: [`nanodb`](/packages/vectordb/nanodb))

<br/>

<a href="https://www.jetson-ai-lab.com/tutorial_nanoowl.html"><img src="https://github.com/NVIDIA-AI-IOT/nanoowl/raw/main/assets/jetson_person_2x.gif"></a>
> [NanoOWL - Open Vocabulary Object Detection ViT](https://www.jetson-ai-lab.com/tutorial_nanoowl.html) (container: [`nanoowl`](/packages/vit/nanoowl))

<a href="https://www.youtube.com/watch?v=w48i8FmVvLA"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava.gif"></a>
> [Live Llava on Jetson AGX Orin](https://youtu.be/X-OXxPiUTuU) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

<a href="https://www.youtube.com/watch?v=wZq7ynbgRoE"><img width="640px" src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava_bear.jpg"></a>
> [Live Llava 2.0 - VILA + Multimodal NanoDB on Jetson Orin](https://youtu.be/X-OXxPiUTuU) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

<a href="https://www.jetson-ai-lab.com/tutorial_slm.html"><img src="https://www.jetson-ai-lab.com/images/slm_console.gif"></a>
> [Small Language Models (SLM) on Jetson Orin Nano](https://www.jetson-ai-lab.com/tutorial_slm.html) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

<a href="https://www.jetson-ai-lab.com/tutorial_nano-vlm.html#video-sequences"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/video_vila_wildfire.gif"></a>
> [Realtime Video Vision/Language Model with VILA1.5-3b](https://www.jetson-ai-lab.com/tutorial_nano-vlm.html#video-sequences) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

## Citation

Please see [CITATION.cff](CITATION.cff) for citation information.
