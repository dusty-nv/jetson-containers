[![a header for a software project about building containers for AI and machine learning](https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/header_blueprint_rainbow.jpg)](https://www.jetson-ai-lab.com)

[![jetson-ai-lab.io status](https://img.shields.io/website?label=jetson-ai-lab.io&url=https%3A%2F%2Fpypi.jetson-ai-lab.io&up_message=up&up_color=brightgreen&down_message=down&down_color=red)](https://pypi.jetson-ai-lab.io)
# CUDA Containers for Edge AI & Robotics

Modular container build system that provides the latest [**AI/ML packages**](https://pypi.jetson-ai-lab.io/) for [NVIDIA Jetson](https://jetson-ai-lab.com) :rocket::robot:

|                       |                                                                                |
|-----------------------|--------------------------------------------------------------------------------|
| **ML**                | [`pytorch`](packages/ml/pytorch) [`tensorflow`](packages/ml/tensorflow) [`jax`](packages/ml/jax) [`onnxruntime`](packages/ml/onnxruntime) [`onnxruntime_genai`](packages/ml/onnxruntime_genai) [`onnx`](packages/ml/onnx) [`CTranslate2`](packages/ml/ctranslate2) [`transformer-engine`](packages/ml/transformer-engine) [`triton`](packages/ml/triton) [`tritonserver`](packages/ml/tritonserver) [`apex`](packages/ml/pytorch/apex) [`torch2trt`](packages/ml/pytorch/torch2trt) [`torch3d`](packages/ml/pytorch/torch3d) [`torchao`](packages/ml/pytorch/torchao) [`torchaudio`](packages/ml/pytorch/torchaudio) [`torchvision`](packages/ml/pytorch/torchvision) [`torchcodec`](packages/ml/pytorch/torchcodec) [`torchsaver`](packages/ml/pytorch/torchsaver) [`torch_tensorrt`](packages/ml/pytorch/torch_tensorrt) [`AIM`](packages/ml/aim) [`TVM`](packages/ml/apache/tvm) [`RAPIDS cuDF`](packages/ml/rapids/cudf) [`RAPIDS cuML`](packages/ml/rapids/cuml) [`RAPIDS RAFT`](packages/ml/rapids/raft) |
| **LLM**              | [`SGLang`](packages/llm/sglang) [`vLLM`](packages/llm/vllm) [`MLC`](packages/llm/mlc) [`AWQ`](packages/llm/awq) [`transformers`](packages/llm/transformers) [`text-generation-webui`](packages/llm/text-generation-webui) [`ollama`](packages/llm/ollama) [`llama.cpp`](packages/llm/llama_cpp) [`llama-factory`](packages/llm/llama-factory) [`exllama`](packages/llm/exllama) [`GPTQModel`](packages/llm/gptqmodel) [`DeepSpeed`](packages/llm/deepspeed) [`bitsandbytes`](packages/llm/bitsandbytes) [`NanoLLM`](packages/llm/nano_llm) [`NeMo`](packages/llm/nemo) [`unsloth`](packages/llm/unsloth) [`ktransformers`](packages/llm/ktransformers) [`dynamo`](packages/llm/dynamo) [`sudonim`](packages/llm/sudonim) [`open-webui`](packages/llm/open-webui) [`litellm`](packages/llm/litellm) [`openai`](packages/llm/openai) [`optimum`](packages/llm/optimum) [`TensorRT-LLM`](packages/llm/tensorrt_optimizer/tensorrt_llm) [`ModelOpt`](packages/llm/tensorrt_optimizer/nvidia-modelopt) [`xgrammar`](packages/llm/xgrammar) [`text-generation-inference`](packages/llm/text-generation-inference) [`minference`](packages/llm/minference) [`hymba`](packages/llm/hymba) [`mistral`](packages/llm/mistral) |
| **VLM**              | [`llava`](packages/vlm/llava) [`llama-vision`](packages/vlm/llama-vision) [`VILA`](packages/vlm/vila) [`VILA Microservice`](packages/vlm/vila-microservice) [`LITA`](packages/vlm/lita) [`Prismatic`](packages/vlm/prismatic) [`xtuner`](packages/vlm/xtuner) [`gemma_vlm`](packages/vlm/gemma_vlm) [`VideoLLaMA`](packages/vlm/videollama) [`MiniGPT4`](packages/vlm/minigpt4) |
| **VIT**              | [`NanoOWL`](packages/vit/nanoowl) [`NanoSAM`](packages/vit/nanosam) [`Segment Anything (SAM)`](packages/vit/sam) [`Track Anything (TAM)`](packages/vit/tam) [`clip_trt`](packages/vit/clip_trt) [`EfficientViT`](packages/vit/efficientvit) [`Sapiens`](packages/vit/sapiens) |
| **Attention**        | [`FlashAttention`](packages/attention/flash-attention) [`xformers`](packages/attention/xformers) [`FlashInfer`](packages/attention/flashinfer) [`SageAttention`](packages/attention/sage-attention) [`SpargeAttention`](packages/attention/sparge-attention) [`ParaAttention`](packages/attention/ParaAttention) [`BlockSparseAttention`](packages/attention/block-sparse-attention) [`FlexPrefill`](packages/attention/flexprefill) [`JVP FlashAttention`](packages/attention/jvp-flash-attention) [`LogLinearAttention`](packages/attention/log-linear-attention) [`RadialAttention`](packages/attention/radial-attention) [`xAttention`](packages/attention/xattention) [`TileLang`](packages/attention/tilelang) [`HuggingFace Kernels`](packages/attention/huggingface_kernels) |
| **RAG**              | [`llama-index`](packages/rag/llama-index) [`langchain`](packages/rag/langchain) [`jetson-copilot`](packages/rag/jetson-copilot) [`NanoDB`](packages/vectordb/nanodb) [`FAISS`](packages/vectordb/faiss) [`faiss_lite`](packages/vectordb/faiss_lite) [`graphiti`](packages/rag/graphiti) [`memvid`](packages/rag/memvid) [`n8n`](packages/rag/n8n) |
| **Robotics**         | [`ROS`](packages/physicalAI/ros) [`LeRobot`](packages/physicalAI/lerobot) [`Isaac ROS`](packages/physicalAI/isaac-ros) [`Isaac-GR00T`](packages/physicalAI/Isaac-GR00T) [`Cosmos`](packages/cv/diffusion/cosmos) [`OpenVLA`](packages/physicalAI/vla/openvla) [`Octo`](packages/physicalAI/vla/octo) [`Crossformer`](packages/physicalAI/vla/crossformer) [`isaac-gr00t (VLA)`](packages/physicalAI/vla/isaac-gr00t) [`RoboPoint`](packages/physicalAI/vla/robopoint) [`openpi`](packages/physicalAI/openpi) [`Protomotions`](packages/physicalAI/protomotions) [`OpenDroneMap`](packages/physicalAI/opendronemap) [`ZED`](packages/hw/zed) |
| **Simulation**       | [`Isaac Sim`](packages/physicalAI/sim/isaac-sim) [`Isaac Lab`](packages/physicalAI/sim/isaac-sim/isaac-lab) [`Genesis`](packages/physicalAI/sim/genesis) [`Habitat Sim`](packages/physicalAI/sim/habitat-sim) [`Newton`](packages/physicalAI/sim/newton) [`MimicGen`](packages/physicalAI/sim/mimicgen) [`MuJoCo`](packages/physicalAI/sim/mujoco) [`PhysX`](packages/physicalAI/sim/physx) [`RoboGen`](packages/physicalAI/sim/robogen) [`RoboMimic`](packages/physicalAI/sim/robomimic) [`RoboSuite`](packages/physicalAI/sim/robosuite) [`Sapien`](packages/physicalAI/sim/sapien) |
| **Diffusion**        | [`Diffusers`](packages/cv/diffusion/diffusers) [`ComfyUI`](packages/cv/diffusion/comfyui) [`FramePack`](packages/cv/diffusion/framepack) [`Self-Forcing`](packages/cv/diffusion/self-forcing) [`AI Toolkit`](packages/cv/diffusion/ai-toolkit) [`Diffusion Policy`](packages/cv/diffusion/diffusion_policy) [`3D Diffusion Policy`](packages/cv/diffusion/3d_diffusion_policy) [`Stable Diffusion WebUI`](packages/cv/diffusion/stable-diffusion-webui) [`SD.Next`](packages/cv/diffusion/sdnext) [`Cache Edit`](packages/cv/diffusion/cache_edit) [`Diffusion C++`](packages/cv/diffusion/diffusion_cpp) |
| **3D Vision**        | [`nerfstudio`](packages/cv/3d/nerf/nerfstudio) [`FruitNeRF`](packages/cv/3d/nerf/fruitnerf) [`gsplat`](packages/cv/3d/gaussian_splatting/gsplat) [`3DGrut`](packages/cv/3d/gaussian_splatting/3dgrut) [`4K4D`](packages/cv/3d/gaussian_splatting/4k4d) [`EasyVolcap`](packages/cv/3d/gaussian_splatting/easyvolcap) [`FastGauss`](packages/cv/3d/gaussian_splatting/fast_gauss) [`NeRFView`](packages/cv/3d/gaussian_splatting/nerfview) [`kaolin`](packages/cv/3d/3dvision/kaolin) [`meshlab`](packages/cv/3d/3dvision/meshlab) [`Open3D`](packages/cv/3d/3dvision/open3d) [`tinycudann`](packages/cv/3d/3dvision/tinycudann) [`hloc`](packages/cv/3d/3dvision/hloc) [`NeRFAcc`](packages/cv/3d/3dvision/nerfacc) [`polyscope`](packages/cv/3d/3dvision/polyscope) [`pycolmap`](packages/cv/3d/3dvision/pycolmap) [`pyceres`](packages/cv/3d/3dvision/pyceres) [`pixsfm`](packages/cv/3d/3dvision/pixsfm) [`pymeshlab`](packages/cv/3d/3dvision/pymeshlab) [`glomap`](packages/cv/3d/3dvision/glomap) [`usdcore`](packages/cv/3d/3dvision/usdcore) [`PartPacker`](packages/cv/3d/3dobjects/partpacker) [`Sparc3D`](packages/cv/3d/3dobjects/sparc3d) |
| **CV**               | [`deepstream`](packages/cv/deepstream) [`holoscan`](packages/cv/holoscan) [`cv-cuda`](packages/cv/cv-cuda) [`opencv`](packages/cv/opencv) [`jetson-inference`](packages/cv/jetson-inference) [`VPI`](packages/cv/vpi) |
| **CUDA**             | [`cuda`](packages/cuda/cuda) [`cuda-python`](packages/cuda/cuda-python) [`pycuda`](packages/cuda/pycuda) [`cudastack`](packages/cuda/cudastack) [`cutlass`](packages/cuda/cutlass) [`cupy`](packages/ml/numeric/cupy) [`numba`](packages/ml/numeric/numba) [`numpy`](packages/ml/numeric/numpy) [`warp`](packages/ml/numeric/warp) [`arrow`](packages/ml/numeric/arrow) |
| **L4T**              | [`l4t-pytorch`](packages/ml/l4t/l4t-pytorch) [`l4t-tensorflow`](packages/ml/l4t/l4t-tensorflow) [`l4t-ml`](packages/ml/l4t/l4t-ml) [`l4t-diffusion`](packages/ml/l4t/l4t-diffusion) [`l4t-text-generation`](packages/ml/l4t/l4t-text-generation) [`l4t-dynamo`](packages/ml/l4t/l4t-dynamo) |
| **Speech**           | [`whisper`](packages/speech/whisper) [`whisper_trt`](packages/speech/whisper_trt) [`whisperx`](packages/speech/whisperx) [`faster-whisper`](packages/speech/faster-whisper) [`piper1-tts`](packages/speech/piper1-tts) [`riva`](packages/speech/riva-client) [`audiocraft`](packages/speech/audiocraft) [`voicecraft`](packages/speech/voicecraft) [`xtts`](packages/speech/xtts) [`kokoro-tts`](packages/speech/kokoro-tts) [`spark-tts`](packages/speech/spark-tts) [`chatterbox-tts`](packages/speech/chatterbox-tts) [`speaches`](packages/speech/speaches) [`voice-pro`](packages/speech/voice-pro) [`espeak`](packages/speech/espeak) |
| **Mamba**            | [`mamba`](packages/ml/mamba/mamba) [`mambavision`](packages/ml/mamba/mambavision) [`cobra`](packages/ml/mamba/cobra) [`dimba`](packages/ml/mamba/dimba) [`videomambasuite`](packages/ml/mamba/videomambasuite) [`zigma`](packages/ml/mamba/zigma) |
| **KANs**             | [`pykan`](packages/ml/kans/pykan) [`kat`](packages/ml/kans/kat) |
| **xLSTM**            | [`xlstm`](packages/ml/xlstm/xlstm) [`mlstm_kernels`](packages/ml/xlstm/mlstm_kernels) [`pltsm`](packages/ml/xlstm/pltsm) |
| **Multimedia**       | [`ffmpeg`](packages/multimedia/ffmpeg) [`gstreamer`](packages/multimedia/gstreamer) [`jetson-utils`](packages/multimedia/jetson-utils) [`decord`](packages/multimedia/decord) [`pyav`](packages/multimedia/pyav) [`opengl`](packages/multimedia/opengl) [`vulkan`](packages/multimedia/vulkan) [`video-codec-sdk`](packages/multimedia/video-codec-sdk) [`libcom`](packages/multimedia/libcom) |
| **Code**             | [`JupyterLab`](packages/code/jupyterlab) [`VSCode`](packages/code/vscode) [`OpenHands`](packages/code/openhands) |
| **Hardware**         | [`ZED`](packages/hw/zed) [`librealsense`](packages/hw/librealsense) [`jetcam`](packages/hw/jetcam) [`canbus`](packages/hw/canbus) [`canable`](packages/hw/canable) [`cangaroo`](packages/hw/cangaroo) [`OLED`](packages/hw/oled) [`PL2303`](packages/hw/pl2303) |
| **Home/IoT**         | [`homeassistant-core`](packages/smart-home/homeassistant-core) [`wyoming-whisper`](packages/smart-home/wyoming/wyoming-whisper) [`wyoming-openwakeword`](packages/smart-home/wyoming/wyoming-openwakeword) [`wyoming-piper`](packages/smart-home/wyoming/wyoming-piper) [`wyoming-assist-microphone`](packages/smart-home/wyoming/wyoming-assist-microphone) |

See the [**`packages`**](packages) directory for the full list, including pre-built container images for JetPack/L4T.

Using the included tools, you can easily combine packages together for building your own containers.  Want to run ROS2 with PyTorch and Transformers?  No problem - just do the [system setup](/docs/setup.md), and build it on your Jetson:

```bash
$ jetson-containers build --name=my_container pytorch transformers ros:humble-desktop
```

There are shortcuts for running containers too - this will pull or build a [`l4t-pytorch`](packages/ml/l4t/l4t-pytorch) image that's compatible:

```bash
$ jetson-containers run $(autotag l4t-pytorch)
```
> <sup>[`jetson-containers run`](/docs/run.md) launches [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some added defaults (like `--runtime nvidia`, mounted `/data` cache and devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

If you look at any package's readme (like [`l4t-pytorch`](packages/ml/l4t/l4t-pytorch)), it will have detailed instructions for running it.

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
> See the **[`Ubuntu 24.04`](/docs/build.md#2404-containers)** section of the docs for details and a list of available containers
> Thanks to all our contributors from **[`Discord`](https://discord.gg/BmqNSK4886)** and AI community for their support


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
