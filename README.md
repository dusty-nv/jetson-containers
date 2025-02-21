[![a header for a software project about building containers for AI and machine learning](https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/header_blueprint_rainbow.jpg)](https://www.jetson-ai-lab.com)

# Machine Learning Containers for Jetson and JetPack

[![l4t-pytorch](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp51.yml?label=l4t-pytorch)](/packages/l4t/l4t-pytorch)  [![l4t-tensorflow](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf2_jp51.yml?label=l4t-tensorflow)](/packages/l4t/l4t-tensorflow) [![l4t-ml](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-ml_jp51.yml?label=l4t-ml)](/packages/l4t/l4t-ml) [![l4t-diffusion](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-diffusion_jp51.yml?label=l4t-diffusion)](/packages/l4t/l4t-diffusion) [![l4t-text-generation](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-text-generation_jp60.yml?label=l4t-text-generation)](/packages/l4t/l4t-text-generation)

Modular container build system that provides the latest [**AI/ML packages**](http://jetson.webredirect.org/) for [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) :rocket::robot:

| |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **ML** | [`pytorch`](packages/pytorch) [`tensorflow`](packages/ml/tensorflow) [`jax`](packages/ml/jax) [`onnxruntime`](packages/ml/onnxruntime) [`deepstream`](packages/multimedia/deepstream) [`holoscan`](packages/multimedia/holoscan) [`CTranslate2`](packages/ml/ctranslate2) [`JupyterLab`](packages/ml/jupyterlab)                                                                                                                                                                                                                                                                               |
| **LLM** | [`NanoLLM`](packages/llm/nano_llm) [`transformers`](packages/llm/transformers) [`text-generation-webui`](packages/llm/text-generation-webui) [`ollama`](packages/llm/ollama) [`llama.cpp`](packages/llm/llama_cpp) [`llama-factory`](packages/llm/llama-factory) [`exllama`](packages/llm/exllama) [`vLLM`](packages/llm/vllm) [`MLC`](packages/llm/mlc) [`AWQ`](packages/llm/awq) [`AutoGPTQ`](packages/llm/auto_gptq) [`FlashAttention`](packages/llm/flash-attention) [`DeepSpeed`](packages/llm/deepspeed) [`bitsandbytes`](packages/llm/bitsandbytes) [`xformers`](packages/llm/xformers) |
| **VLM** | [`llava`](packages/vlm/llava) [`llama-vision`](packages/vlm/llama-vision) [`VILA`](packages/vlm/vila) [`LITA`](packages/vlm/lita) [`NanoLLM`](packages/llm/nano_llm) [`ShapeLLM`](packages/vlm/shape-llm) [`Prismatic`](packages/vlm/prismatic) [`xtuner`](packages/vlm/xtuner)                                                                                                                                                                                                                                                                                                                |
| **VIT** | [`NanoOWL`](packages/vit/nanoowl) [`NanoSAM`](packages/vit/nanosam) [`Segment Anything (SAM)`](packages/vit/sam) [`Track Anything (TAM)`](packages/vit/tam) [`clip_trt`](packages/vit/clip_trt)                                                                                                                                                                                                                                                                                                                                                                                                |
| **RAG** | [`llama-index`](packages/rag/llama-index) [`langchain`](packages/rag/langchain) [`jetson-copilot`](packages/rag/jetson-copilot) [`NanoDB`](packages/vectordb/nanodb) [`FAISS`](packages/vectordb/faiss) [`RAFT`](packages/ml/rapids/raft)                                                                                                                                                                                                                                                                                                                                                      |
| **L4T** | [`l4t-pytorch`](packages/l4t/l4t-pytorch) [`l4t-tensorflow`](packages/l4t/l4t-tensorflow) [`l4t-ml`](packages/l4t/l4t-ml) [`l4t-diffusion`](packages/l4t/l4t-diffusion) [`l4t-text-generation`](packages/l4t/l4t-text-generation)                                                                                                                                                                                                                                                                                                                                                              |
| **CUDA** | [`cupy`](packages/cuda/cupy) [`cuda-python`](packages/cuda/cuda-python) [`pycuda`](packages/cuda/pycuda) [`numba`](packages/numba) [`opencv:cuda`](packages/opencv) [`cudf`](packages/ml/rapids/cudf) [`cuml`](packages/ml/rapids/cuml)                                                                                                                                                                                                                                                                                                                                                        |
| **Robotics** | [`ROS`](packages/ros) [`LeRobot`](packages/robots/lerobot) [`OpenVLA`](packages/robots/openvla) [`3D Diffusion Policy`](packages/robots/3d_diffusion_policy) [`Crossformer`](packages/robots/crossformer) [`MimicGen`](packages/robots/mimicgen) [`OpenDroneMap`](packages/robots/opendronemap) [`ZED`](packages/hardware/zed)                                                                                                                                                                                                                                                                 |
| **Graphics** | [`Cosmos`](packages/robots/cosmos) [`stable-diffusion-webui`](packages/diffusion/stable-diffusion-webui) [`comfyui`](packages/diffusion/comfyui) [`nerfstudio`](packages/nerf/nerfstudio) [`meshlab`](packages/nerf/meshlab) [`pixsfm`](packages/nerf/pixsfm) [`gsplat`](packages/nerf/gsplat)                                                                                                                                                                                                                                                                                              |
| **Mamba** | [`mamba`](packages/mamba/mamba) [`mambavision`](packages/mamba/mambavision) [`cobra`](packages/mamba/cobra) [`dimba`](packages/mamba/dimba) [`videomambasuite`](packages/mamba/videomambasuite)                                                                                                                                                                                                                                                                                                                                                                                                |
| **Speech** | [`whisper`](packages/speech/whisper) [`whisper_trt`](packages/speech/whisper_trt) [`piper`](packages/speech/piper-tts) [`riva`](packages/speech/riva-client) [`audiocraft`](packages/speech/audiocraft) [`voicecraft`](packages/speech/voicecraft) [`xtts`](packages/speech/xtts)                                                                                                                                                                                                                                                                                                              |
| **Home/IoT** | [`homeassistant-core`](packages/smart-home/homeassistant-core) [`wyoming-whisper`](packages/smart-home/wyoming/wyoming-whisper) [`wyoming-openwakeword`](packages/smart-home/wyoming/openwakeword) [`wyoming-piper`](packages/smart-home/wyoming/piper)                                                                                                                                                                                                                                                                                                                                        |

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
CUDA_VERSION=12.4 jetson-containers build transformers
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

## Gallery

<a href="https://www.youtube.com/watch?v=UOjqF3YCGkY"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_llava_clip.gif"></a>
> [Multimodal Voice Chat with LLaVA-1.5 13B on NVIDIA Jetson AGX Orin](https://www.youtube.com/watch?v=9ObzbbBTbcc) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))   

<br/>

<a href="https://www.youtube.com/watch?v=hswNSZTvEFE"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_70b_yt.jpg" width="800px"></a>
> [Interactive Voice Chat with Llama-2-70B on NVIDIA Jetson AGX Orin](https://www.youtube.com/watch?v=wzLHAgDxMjQ) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))  

<br/>

<a href="https://www.youtube.com/watch?v=OJT-Ax0CkhU"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/nanodb_tennis.jpg"></a>
> [Realtime Multimodal VectorDB on NVIDIA Jetson](https://www.youtube.com/watch?v=wzLHAgDxMjQ) (container: [`nanodb`](/packages/vectordb/nanodb))  

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
  
