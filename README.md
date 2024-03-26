![a header for a software project about building containers for AI and machine learning](https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/header.jpg)

# Machine Learning Containers for Jetson and JetPack

[![l4t-pytorch](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp51.yml?label=l4t-pytorch)](/packages/l4t/l4t-pytorch)  [![l4t-tensorflow](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf2_jp51.yml?label=l4t-tensorflow)](/packages/l4t/l4t-tensorflow) [![l4t-ml](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-ml_jp51.yml?label=l4t-ml)](/packages/l4t/l4t-ml) [![l4t-diffusion](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-diffusion_jp51.yml?label=l4t-diffusion)](/packages/l4t/l4t-diffusion) [![l4t-text-generation](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-text-generation_jp60.yml?label=l4t-text-generation)](/packages/l4t/l4t-text-generation)

Modular container build system that provides various [**AI/ML packages**](packages) for [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) :rocket::robot:

| | |
|---|---|
| **ML** | [`pytorch`](packages/pytorch) [`tensorflow`](packages/tensorflow) [`onnxruntime`](packages/onnxruntime) [`deepstream`](packages/deepstream) [`tritonserver`](packages/tritonserver) [`jupyterlab`](packages/jupyterlab) [`stable-diffusion`](packages/diffusion/stable-diffusion-webui) |
| **LLM** | [`transformers`](packages/llm/transformers) [`text-generation-webui`](packages/llm/text-generation-webui) [`llava`](packages/llm/llava) [`llama.cpp`](packages/llm/llama_cpp) [`exllama`](packages/llm/exllama) [`llamaspeak`](packages/llm/llamaspeak) [`local_llm`](packages/llm/local_llm) [`awq`](packages/llm/awq) [`AutoGPTQ`](packages/llm/auto_gptq) [`MiniGPT-4`](packages/llm/minigpt4) [`MLC`](packages/llm/mlc) [`langchain`](packages/llm/langchain) [`optimum`](packages/llm/optimum) [`nemo`](packages/nemo) |
| **L4T** | [`l4t-pytorch`](packages/l4t/l4t-pytorch) [`l4t-tensorflow`](packages/l4t/l4t-tensorflow) [`l4t-ml`](packages/l4t/l4t-ml) [`l4t-diffusion`](packages/l4t/l4t-diffusion) [`l4t-text-generation`](packages/l4t/l4t-text-generation) |
| **VIT** | [`NanoOWL`](packages/vit/nanoowl) [`NanoSAM`](packages/vit/nanosam) [`Segment Anything (SAM)`](packages/vit/sam) [`Track Anything (TAM)`](packages/vit/tam) |
| **CUDA** | [`cupy`](packages/cuda/cupy) [`cuda-python`](packages/cuda/cuda-python) [`pycuda`](packages/cuda/pycuda) [`numba`](packages/numba) [`cudf`](packages/rapids/cudf) [`cuml`](packages/rapids/cuml) |
| **Robotics** | [`ros`](packages/ros) [`ros2`](packages/ros) [`opencv:cuda`](packages/opencv) [`realsense`](packages/realsense) [`zed`](packages/zed) |
| **VectorDB** | [`NanoDB`](packages/vectordb/nanodb) [`FAISS`](packages/vectordb/faiss) [`RAFT`](packages/rapids/raft) |
| **Audio** | [`whisper`](packages/audio/whisper) [`whisperX`](packages/audio/whisperx) [`riva`](packages/audio/riva-client) [`audiocraft`](packages/audio/audiocraft) |

See the [**`packages`**](packages) directory for the full list, including pre-built container images and CI/CD status for JetPack/L4T.

Using the included tools, you can easily combine packages together for building your own containers.  Want to run ROS2 with PyTorch and Transformers?  No problem - just do the [system setup](/docs/setup.md), and build it on your Jetson like this:

```bash
$ ./build.sh --name=my_container pytorch transformers ros:humble-desktop
```

There are shortcuts for running containers too - this will pull or build a [`l4t-pytorch`](packages/l4t/l4t-pytorch) image that's compatible:

```bash
$ ./run.sh $(./autotag l4t-pytorch)
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

If you look at any package's readme (like [`l4t-pytorch`](packages/l4t/l4t-pytorch)), it will have detailed instructions for running it's container.

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
sudo apt-get update && sudo apt-get install git python3-pip
git clone --depth=1 https://github.com/dusty-nv/jetson-containers
cd jetson-containers
pip3 install -r requirements.txt
./run.sh $(./autotag l4t-pytorch)
```

Or you can manually run a [container image](https://hub.docker.com/r/dustynv) of your choice without using the helper scripts above:

```bash
sudo docker run --runtime nvidia -it --rm --network=host dustynv/l4t-pytorch:r35.4.1
```

Looking for the old jetson-containers?   See the [`legacy`](https://github.com/dusty-nv/jetson-containers/tree/legacy) branch.

## Gallery

<a href="https://www.youtube.com/watch?v=9ObzbbBTbcc"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_llava_clip.gif"></a>
> [Multimodal Voice Chat with LLaVA-1.5 13B on NVIDIA Jetson AGX Orin](https://www.youtube.com/watch?v=9ObzbbBTbcc) (container: [`local_llm`](/packages/llm/local_llm))  

<br/>

<a href="https://www.youtube.com/watch?v=wzLHAgDxMjQ"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_70b_yt.jpg" width="800px"></a>
> [Interactive Voice Chat with Llama-2-70B on NVIDIA Jetson AGX Orin](https://www.youtube.com/watch?v=wzLHAgDxMjQ) (container: [`local_llm`](/packages/llm/local_llm))  

<br/>

<a href="https://youtu.be/ayqKpQNd1Jw"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/nanodb_tennis.jpg"></a>
> [Realtime Multimodal VectorDB on NVIDIA Jetson](https://www.youtube.com/watch?v=wzLHAgDxMjQ) (container: [`nanodb`](/packages/vectordb/nanodb))  

<br/>

<a href="https://www.jetson-ai-lab.com/tutorial_nanoowl.html"><img src="https://github.com/NVIDIA-AI-IOT/nanoowl/raw/main/assets/jetson_person_2x.gif"></a>
> [NanoOWL - Open Vocabulary Object Detection ViT](https://www.jetson-ai-lab.com/tutorial_nanoowl.html) (container: [`nanoowl`](/packages/vit/nanoowl))  

<a href="https://youtu.be/X-OXxPiUTuU"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava.gif"></a>
> [Live Llava on Jetson AGX Orin](https://youtu.be/X-OXxPiUTuU) (container: [`local_llm`](/packages/llm/local_llm#live-llava)) 

<a href="https://youtu.be/dRmAGGuupuE"><img width="640px" src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava_bear.jpg"></a>
> [Live Llava 2.0 - VILA + Multimodal NanoDB on Jetson Orin](https://youtu.be/X-OXxPiUTuU) (container: [`local_llm`](/packages/llm/local_llm#live-llava)) 

<a href="https://www.jetson-ai-lab.com/tutorial_slm.html"><img src="https://www.jetson-ai-lab.com/images/slm_console.gif"></a>
> [Small Language Models (SLM) on Jetson Orin Nano](https://www.jetson-ai-lab.com/tutorial_slm.html) (container: [`local_llm`](/packages/llm/local_llm#live-llava)) 