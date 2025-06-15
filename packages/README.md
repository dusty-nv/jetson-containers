# Packages
> [`AUDIO`](#user-content-audio) [`BUILD`](#user-content-build) [`CORE`](#user-content-core) [`CUDA`](#user-content-cuda) [`DIFFUSION`](#user-content-diffusion) [`HARDWARE`](#user-content-hardware) [`JAX`](#user-content-jax) [`LLM`](#user-content-llm) [`MAMBA`](#user-content-mamba) [`ML`](#user-content-ml) [`MULTIMEDIA`](#user-content-multimedia) [`NERF`](#user-content-nerf) [`OTHER`](#user-content-other) [`PYTORCH`](#user-content-pytorch) [`RAG`](#user-content-rag) [`RAPIDS`](#user-content-rapids) [`ROBOTS`](#user-content-robots) [`ROS`](#user-content-ros) [`SIM`](#user-content-sim) [`SMART-HOME`](#user-content-smart-home) [`TRANSFORMER`](#user-content-transformer) [`VECTORDB`](#user-content-vectordb) [`VIT`](#user-content-vit) [`WYOMING`](#user-content-wyoming) 

|            |            |
|------------|------------|
| <a id="audio">**`AUDIO`**</a> | |
| &nbsp;&nbsp;&nbsp; [`audiocraft`](/packages/speech/audiocraft) | [![`audiocraft_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/audiocraft_jp51.yml?label=audiocraft:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/audiocraft_jp51.yml) [![`audiocraft_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/audiocraft_jp60.yml?label=audiocraft:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/audiocraft_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`faster-whisper`](/packages/speech/faster-whisper) |  |
| &nbsp;&nbsp;&nbsp; [`piper-tts`](/packages/speech/piper-tts) |  |
| &nbsp;&nbsp;&nbsp; [`riva-client:cpp`](/packages/speech/riva-client) | [![`riva-client-cpp_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/riva-client-cpp_jp51.yml?label=riva-client-cpp:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/riva-client-cpp_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`riva-client:python`](/packages/speech/riva-client) | [![`riva-client-python_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/riva-client-python_jp51.yml?label=riva-client-python:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/riva-client-python_jp51.yml) [![`riva-client-python_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/riva-client-python_jp60.yml?label=riva-client-python:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/riva-client-python_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`voicecraft`](/packages/speech/voicecraft) |  |
| &nbsp;&nbsp;&nbsp; [`whisper`](/packages/speech/whisper) | [![`whisper_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/whisper_jp51.yml?label=whisper:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/whisper_jp51.yml) [![`whisper_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/whisper_jp60.yml?label=whisper:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/whisper_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`whisper_trt`](/packages/speech/whisper_trt) |  |
| &nbsp;&nbsp;&nbsp; [`whisperx`](/packages/speech/whisperx) | [![`whisperx_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/whisperx_jp51.yml?label=whisperx:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/whisperx_jp51.yml) [![`whisperx_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/whisperx_jp60.yml?label=whisperx:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/whisperx_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`xtts`](/packages/speech/xtts) |  |
| <a id="build">**`BUILD`**</a> | |
| &nbsp;&nbsp;&nbsp; [`bazel`](/packages/build/bazel) | [![`bazel_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/bazel_jp46.yml?label=bazel:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/bazel_jp46.yml) [![`bazel_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/bazel_jp51.yml?label=bazel:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/bazel_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`build-essential`](/packages/build/build-essential) | [![`build-essential_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/build-essential_jp46.yml?label=build-essential:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/build-essential_jp46.yml) [![`build-essential_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/build-essential_jp51.yml?label=build-essential:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/build-essential_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`cmake:apt`](/packages/build/cmake/cmake_apt) | [![`cmake-apt_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cmake-apt_jp46.yml?label=cmake-apt:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cmake-apt_jp46.yml) [![`cmake-apt_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cmake-apt_jp51.yml?label=cmake-apt:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cmake-apt_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`cmake:pip`](/packages/build/cmake/cmake_pip) | [![`cmake-pip_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cmake-pip_jp46.yml?label=cmake-pip:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cmake-pip_jp46.yml) [![`cmake-pip_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cmake-pip_jp51.yml?label=cmake-pip:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cmake-pip_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`nodejs`](/packages/build/nodejs) |  |
| &nbsp;&nbsp;&nbsp; [`pip_cache:cu122`](/packages/cuda/cuda) |  |
| &nbsp;&nbsp;&nbsp; [`pip_cache:cu124`](/packages/cuda/cuda) |  |
| &nbsp;&nbsp;&nbsp; [`pip_cache:cu126`](/packages/cuda/cuda) |  |
| &nbsp;&nbsp;&nbsp; [`protobuf:apt`](/packages/build/protobuf/protobuf_apt) | [![`protobuf-apt_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/protobuf-apt_jp46.yml?label=protobuf-apt:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/protobuf-apt_jp46.yml) [![`protobuf-apt_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/protobuf-apt_jp51.yml?label=protobuf-apt:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/protobuf-apt_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`protobuf:cpp`](/packages/build/protobuf/protobuf_cpp) | [![`protobuf-cpp_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/protobuf-cpp_jp46.yml?label=protobuf-cpp:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/protobuf-cpp_jp46.yml) [![`protobuf-cpp_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/protobuf-cpp_jp51.yml?label=protobuf-cpp:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/protobuf-cpp_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`python:3.10`](/packages/build/python) |  |
| &nbsp;&nbsp;&nbsp; [`python:3.11`](/packages/build/python) |  |
| &nbsp;&nbsp;&nbsp; [`python:3.12`](/packages/build/python) |  |
| &nbsp;&nbsp;&nbsp; [`python:3.13`](/packages/build/python) |  |
| &nbsp;&nbsp;&nbsp; [`rust`](/packages/build/rust) | [![`rust_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/rust_jp46.yml?label=rust:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/rust_jp46.yml) [![`rust_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/rust_jp51.yml?label=rust:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/rust_jp51.yml) |
| <a id="core">**`CORE`**</a> | |
| &nbsp;&nbsp;&nbsp; [`arrow:12.0.1`](/packages/numeric/arrow) |  |
| &nbsp;&nbsp;&nbsp; [`arrow:14.0.1`](/packages/numeric/arrow) |  |
| &nbsp;&nbsp;&nbsp; [`arrow:5.0.0`](/packages/numeric/arrow) |  |
| &nbsp;&nbsp;&nbsp; [`docker`](/packages/build/docker) |  |
| &nbsp;&nbsp;&nbsp; [`h5py`](/packages/build/h5py) |  |
| &nbsp;&nbsp;&nbsp; [`jupyterlab`](/packages/jupyterlab) | [![`jupyterlab_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jupyterlab_jp46.yml?label=jupyterlab:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jupyterlab_jp46.yml) [![`jupyterlab_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jupyterlab_jp51.yml?label=jupyterlab:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jupyterlab_jp51.yml) [![`jupyterlab_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jupyterlab_jp60.yml?label=jupyterlab:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jupyterlab_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`jupyterlab:myst`](/packages/jupyterlab) |  |
| &nbsp;&nbsp;&nbsp; [`ninja`](/packages/build/ninja) |  |
| &nbsp;&nbsp;&nbsp; [`numpy`](/packages/numeric/numpy) | [![`numpy_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/numpy_jp46.yml?label=numpy:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/numpy_jp46.yml) [![`numpy_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/numpy_jp51.yml?label=numpy:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/numpy_jp51.yml) [![`numpy_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/numpy_jp60.yml?label=numpy:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/numpy_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`opencv:4.11.0`](/packages/opencv) |  |
| &nbsp;&nbsp;&nbsp; [`opencv:4.11.0-builder`](/packages/opencv) |  |
| &nbsp;&nbsp;&nbsp; [`opencv:4.11.0-meta`](/packages/opencv) |  |
| &nbsp;&nbsp;&nbsp; [`opencv:4.8.1`](/packages/opencv) | [![`opencv-481_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/opencv-481_jp60.yml?label=opencv-481:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/opencv-481_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`opencv:4.8.1-builder`](/packages/opencv) |  |
| &nbsp;&nbsp;&nbsp; [`opencv:4.8.1-deb`](/packages/opencv) |  |
| &nbsp;&nbsp;&nbsp; [`opencv:4.8.1-meta`](/packages/opencv) |  |
| <a id="cuda">**`CUDA`**</a> | |
| &nbsp;&nbsp;&nbsp; [`cuda-python:12.2`](/packages/cuda/cuda-python) |  |
| &nbsp;&nbsp;&nbsp; [`cuda-python:12.4`](/packages/cuda/cuda-python) |  |
| &nbsp;&nbsp;&nbsp; [`cuda-python:12.6`](/packages/cuda/cuda-python) |  |
| &nbsp;&nbsp;&nbsp; [`cuda:12.2`](/packages/cuda/cuda) | [![`cuda-122_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cuda-122_jp60.yml?label=cuda-122:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cuda-122_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`cuda:12.2-samples`](/packages/cuda/cuda) | [![`cuda-122-samples_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cuda-122-samples_jp60.yml?label=cuda-122-samples:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cuda-122-samples_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`cuda:12.4`](/packages/cuda/cuda) |  |
| &nbsp;&nbsp;&nbsp; [`cuda:12.4-samples`](/packages/cuda/cuda) |  |
| &nbsp;&nbsp;&nbsp; [`cuda:12.6`](/packages/cuda/cuda) |  |
| &nbsp;&nbsp;&nbsp; [`cuda:12.6-samples`](/packages/cuda/cuda) |  |
| &nbsp;&nbsp;&nbsp; [`cudnn:8.9`](/packages/cuda/cudnn) | [![`cudnn-89_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cudnn-89_jp60.yml?label=cudnn-89:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cudnn-89_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`cudnn:9.0`](/packages/cuda/cudnn) |  |
| &nbsp;&nbsp;&nbsp; [`cudnn:9.3`](/packages/cuda/cudnn) |  |
| &nbsp;&nbsp;&nbsp; [`cupy`](/packages/cuda/cupy) | [![`cupy_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cupy_jp46.yml?label=cupy:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cupy_jp46.yml) [![`cupy_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cupy_jp51.yml?label=cupy:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cupy_jp51.yml) [![`cupy_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cupy_jp60.yml?label=cupy:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cupy_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`jetson-utils`](/packages/jetson-inference/jetson-utils) | [![`jetson-utils_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jetson-utils_jp46.yml?label=jetson-utils:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jetson-utils_jp46.yml) [![`jetson-utils_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jetson-utils_jp51.yml?label=jetson-utils:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jetson-utils_jp51.yml) [![`jetson-utils_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jetson-utils_jp60.yml?label=jetson-utils:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jetson-utils_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`numba`](/packages/numeric/numba) | [![`numba_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/numba_jp46.yml?label=numba:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/numba_jp46.yml) [![`numba_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/numba_jp51.yml?label=numba:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/numba_jp51.yml) [![`numba_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/numba_jp60.yml?label=numba:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/numba_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`pycuda`](/packages/cuda/pycuda) | [![`pycuda_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/pycuda_jp46.yml?label=pycuda:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/pycuda_jp46.yml) [![`pycuda_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/pycuda_jp51.yml?label=pycuda:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/pycuda_jp51.yml) [![`pycuda_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/pycuda_jp60.yml?label=pycuda:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/pycuda_jp60.yml) |
| <a id="diffusion">**`DIFFUSION`**</a> | |
| &nbsp;&nbsp;&nbsp; [`ai-toolkit`](/packages/diffusion/ai-toolkit) |  |
| &nbsp;&nbsp;&nbsp; [`comfyui`](/packages/diffusion/comfyui) |  |
| &nbsp;&nbsp;&nbsp; [`diffusers:0.30.2`](/packages/diffusion/diffusers) |  |
| &nbsp;&nbsp;&nbsp; [`diffusers:0.30.2-builder`](/packages/diffusion/diffusers) |  |
| &nbsp;&nbsp;&nbsp; [`l4t-diffusion`](/packages/l4t/l4t-diffusion) | [![`l4t-diffusion_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-diffusion_jp51.yml?label=l4t-diffusion:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-diffusion_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`stable-diffusion`](/packages/diffusion/stable-diffusion) | [![`stable-diffusion_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/stable-diffusion_jp51.yml?label=stable-diffusion:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/stable-diffusion_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) | [![`stable-diffusion-webui_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/stable-diffusion-webui_jp51.yml?label=stable-diffusion-webui:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/stable-diffusion-webui_jp51.yml) [![`stable-diffusion-webui_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/stable-diffusion-webui_jp60.yml?label=stable-diffusion-webui:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/stable-diffusion-webui_jp60.yml) |
| <a id="hardware">**`HARDWARE`**</a> | |
| &nbsp;&nbsp;&nbsp; [`jetcam`](/packages/hardware/jetcam) |  |
| &nbsp;&nbsp;&nbsp; [`jupyter_clickable_image_widget`](/packages/hardware/jupyter_clickable_image_widget) |  |
| &nbsp;&nbsp;&nbsp; [`oled`](/packages/hardware/oled) |  |
| &nbsp;&nbsp;&nbsp; [`realsense`](/packages/hardware/realsense) | [![`realsense_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/realsense_jp46.yml?label=realsense:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/realsense_jp46.yml) [![`realsense_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/realsense_jp51.yml?label=realsense:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/realsense_jp51.yml) [![`realsense_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/realsense_jp60.yml?label=realsense:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/realsense_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`zed`](/packages/hardware/zed) | [![`zed_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/zed_jp46.yml?label=zed:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/zed_jp46.yml) [![`zed_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/zed_jp51.yml?label=zed:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/zed_jp51.yml) |
| <a id="jax">**`JAX`**</a> | |
| &nbsp;&nbsp;&nbsp; [`jax:0.4.28`](/packages/jax) |  |
| &nbsp;&nbsp;&nbsp; [`jax:0.4.28-builder`](/packages/jax) |  |
| &nbsp;&nbsp;&nbsp; [`jax:0.4.30`](/packages/jax) |  |
| &nbsp;&nbsp;&nbsp; [`jax:0.4.30-builder`](/packages/jax) |  |
| &nbsp;&nbsp;&nbsp; [`jax:0.4.32`](/packages/jax) |  |
| &nbsp;&nbsp;&nbsp; [`jax:0.4.32-builder`](/packages/jax) |  |
| <a id="llm">**`LLM`**</a> | |
| &nbsp;&nbsp;&nbsp; [`auto_awq:0.2.4`](/packages/llm/auto_awq) |  |
| &nbsp;&nbsp;&nbsp; [`auto_gptq:0.7.1`](/packages/llm/auto_gptq) |  |
| &nbsp;&nbsp;&nbsp; [`awq:0.1.0`](/packages/llm/awq) |  |
| &nbsp;&nbsp;&nbsp; [`awq:0.1.0-builder`](/packages/llm/awq) |  |
| &nbsp;&nbsp;&nbsp; [`bitsandbytes:0.43.3`](/packages/llm/bitsandbytes) |  |
| &nbsp;&nbsp;&nbsp; [`bitsandbytes:0.43.3-builder`](/packages/llm/bitsandbytes) |  |
| &nbsp;&nbsp;&nbsp; [`exllama:0.0.15`](/packages/llm/exllama) |  |
| &nbsp;&nbsp;&nbsp; [`flash-attention:2.5.7`](/packages/attention/flash-attention) |  |
| &nbsp;&nbsp;&nbsp; [`flash-attention:2.5.7-builder`](/packages/attention/flash-attention) |  |
| &nbsp;&nbsp;&nbsp; [`flash-attention:2.6.3`](/packages/attention/flash-attention) |  |
| &nbsp;&nbsp;&nbsp; [`flash-attention:2.6.3-builder`](/packages/attention/flash-attention) |  |
| &nbsp;&nbsp;&nbsp; [`gptq-for-llama`](/packages/llm/gptq-for-llama) | [![`gptq-for-llama_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/gptq-for-llama_jp51.yml?label=gptq-for-llama:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/gptq-for-llama_jp51.yml) [![`gptq-for-llama_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/gptq-for-llama_jp60.yml?label=gptq-for-llama:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/gptq-for-llama_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`huggingface_hub`](/packages/llm/huggingface_hub) | [![`huggingface_hub_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/huggingface_hub_jp46.yml?label=huggingface_hub:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/huggingface_hub_jp46.yml) [![`huggingface_hub_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/huggingface_hub_jp51.yml?label=huggingface_hub:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/huggingface_hub_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`l4t-text-generation`](/packages/l4t/l4t-text-generation) | [![`l4t-text-generation_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-text-generation_jp51.yml?label=l4t-text-generation:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-text-generation_jp51.yml) [![`l4t-text-generation_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-text-generation_jp60.yml?label=l4t-text-generation:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-text-generation_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`llama-factory`](/packages/llm/llama-factory) |  |
| &nbsp;&nbsp;&nbsp; [`llama_cpp:0.2.57`](/packages/llm/llama_cpp) |  |
| &nbsp;&nbsp;&nbsp; [`llama_cpp:0.2.70`](/packages/llm/llama_cpp) |  |
| &nbsp;&nbsp;&nbsp; [`llama_cpp:0.2.83`](/packages/llm/llama_cpp) |  |
| &nbsp;&nbsp;&nbsp; [`llamaspeak`](/packages/llm/llamaspeak) | [![`llamaspeak_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/llamaspeak_jp51.yml?label=llamaspeak:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/llamaspeak_jp51.yml) [![`llamaspeak_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/llamaspeak_jp60.yml?label=llamaspeak:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/llamaspeak_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`llava`](/packages/llm/llava) | [![`llava_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/llava_jp51.yml?label=llava:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/llava_jp51.yml) [![`llava_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/llava_jp60.yml?label=llava:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/llava_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`local_llm`](/packages/llm/local_llm) | [![`local_llm_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/local_llm_jp51.yml?label=local_llm:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/local_llm_jp51.yml) [![`local_llm_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/local_llm_jp60.yml?label=local_llm:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/local_llm_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`minigpt4`](/packages/llm/minigpt4) | [![`minigpt4_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/minigpt4_jp51.yml?label=minigpt4:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/minigpt4_jp51.yml) [![`minigpt4_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/minigpt4_jp60.yml?label=minigpt4:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/minigpt4_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`mlc:0.1.0`](/packages/llm/mlc) |  |
| &nbsp;&nbsp;&nbsp; [`mlc:0.1.0-builder`](/packages/llm/mlc) |  |
| &nbsp;&nbsp;&nbsp; [`mlc:0.1.1`](/packages/llm/mlc) |  |
| &nbsp;&nbsp;&nbsp; [`mlc:0.1.1-builder`](/packages/llm/mlc) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.4`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.4-humble`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.4-iron`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.4.1`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.5`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.5-humble`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.5-iron`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.5.1`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.6`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.6-humble`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.6-iron`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.7`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.7-humble`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:24.7-iron`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:main`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:main-foxy`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:main-galactic`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:main-humble`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`nano_llm:main-iron`](/packages/llm/nano_llm) |  |
| &nbsp;&nbsp;&nbsp; [`ollama`](/packages/llm/ollama) |  |
| &nbsp;&nbsp;&nbsp; [`openai`](/packages/llm/openai) |  |
| &nbsp;&nbsp;&nbsp; [`openvla`](/packages/llm/openvla) |  |
| &nbsp;&nbsp;&nbsp; [`openvla:mimicgen`](/packages/llm/openvla) |  |
| &nbsp;&nbsp;&nbsp; [`optimum`](/packages/llm/optimum) | [![`optimum_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/optimum_jp46.yml?label=optimum:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/optimum_jp46.yml) [![`optimum_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/optimum_jp51.yml?label=optimum:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/optimum_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`tensorrt_llm:0.5`](/packages/llm/tensorrt_optimizer/tensorrt_llm) |  |
| &nbsp;&nbsp;&nbsp; [`tensorrt_llm:0.5-builder`](/packages/llm/tensorrt_optimizer/tensorrt_llm) |  |
| &nbsp;&nbsp;&nbsp; [`text-generation-inference`](/packages/llm/text-generation-inference) | [![`text-generation-inference_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/text-generation-inference_jp51.yml?label=text-generation-inference:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/text-generation-inference_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) |  |
| &nbsp;&nbsp;&nbsp; [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) |  |
| &nbsp;&nbsp;&nbsp; [`text-generation-webui:main`](/packages/llm/text-generation-webui) |  |
| &nbsp;&nbsp;&nbsp; [`transformers`](/packages/llm/transformers) | [![`transformers_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/transformers_jp46.yml?label=transformers:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/transformers_jp46.yml) [![`transformers_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/transformers_jp51.yml?label=transformers:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/transformers_jp51.yml) [![`transformers_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/transformers_jp60.yml?label=transformers:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/transformers_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`transformers:git`](/packages/llm/transformers) | [![`transformers-git_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/transformers-git_jp51.yml?label=transformers-git:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/transformers-git_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`transformers:nvgpt`](/packages/llm/transformers) | [![`transformers-nvgpt_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/transformers-nvgpt_jp51.yml?label=transformers-nvgpt:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/transformers-nvgpt_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`xformers:0.0.27.post2`](/packages/llm/xformers) |  |
| &nbsp;&nbsp;&nbsp; [`xformers:0.0.27.post2-builder`](/packages/llm/xformers) |  |
| <a id="mamba">**`MAMBA`**</a> | |
| &nbsp;&nbsp;&nbsp; [`causalconv1d:1.4.0`](/packages/mamba/causalconv1d) |  |
| &nbsp;&nbsp;&nbsp; [`cobra:0.0.1`](/packages/mamba/cobra) |  |
| &nbsp;&nbsp;&nbsp; [`mamba:2.2.2`](/packages/mamba/mamba) |  |
| &nbsp;&nbsp;&nbsp; [`mambavision:1.0`](/packages/mamba/mambavision) |  |
| &nbsp;&nbsp;&nbsp; [`videomambasuite:1.0`](/packages/mamba/videomambasuite) |  |
| <a id="ml">**`ML`**</a> | |
| &nbsp;&nbsp;&nbsp; [`deepstream`](/packages/multimedia/deepstream) | [![`deepstream_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/deepstream_jp46.yml?label=deepstream:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/deepstream_jp46.yml) [![`deepstream_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/deepstream_jp51.yml?label=deepstream:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/deepstream_jp51.yml) [![`deepstream_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/deepstream_jp60.yml?label=deepstream:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/deepstream_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`holoscan`](/packages/holoscan) |  |
| &nbsp;&nbsp;&nbsp; [`jetson-inference:foxy`](/packages/jetson-inference) |  |
| &nbsp;&nbsp;&nbsp; [`jetson-inference:galactic`](/packages/jetson-inference) |  |
| &nbsp;&nbsp;&nbsp; [`jetson-inference:humble`](/packages/jetson-inference) |  |
| &nbsp;&nbsp;&nbsp; [`jetson-inference:iron`](/packages/jetson-inference) |  |
| &nbsp;&nbsp;&nbsp; [`jetson-inference:jazzy`](/packages/jetson-inference) |  |
| &nbsp;&nbsp;&nbsp; [`jetson-inference:main`](/packages/jetson-inference) |  |
| &nbsp;&nbsp;&nbsp; [`l4t-ml`](/packages/l4t/l4t-ml) | [![`l4t-ml_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-ml_jp46.yml?label=l4t-ml:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-ml_jp46.yml) [![`l4t-ml_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-ml_jp51.yml?label=l4t-ml:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-ml_jp51.yml) [![`l4t-ml_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-ml_jp60.yml?label=l4t-ml:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-ml_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`l4t-pytorch`](/packages/l4t/l4t-pytorch) | [![`l4t-pytorch_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp46.yml?label=l4t-pytorch:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-pytorch_jp46.yml) [![`l4t-pytorch_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp51.yml?label=l4t-pytorch:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-pytorch_jp51.yml) [![`l4t-pytorch_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp60.yml?label=l4t-pytorch:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-pytorch_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) | [![`l4t-tensorflow-tf2_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf2_jp46.yml?label=l4t-tensorflow-tf2:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf2_jp46.yml) [![`l4t-tensorflow-tf2_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf2_jp51.yml?label=l4t-tensorflow-tf2:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf2_jp51.yml) [![`l4t-tensorflow-tf2_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf2_jp60.yml?label=l4t-tensorflow-tf2:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-tensorflow-tf2_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`nemo`](/packages/llm/nemo) | [![`nemo_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/nemo_jp46.yml?label=nemo:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/nemo_jp46.yml) [![`nemo_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/nemo_jp51.yml?label=nemo:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/nemo_jp51.yml) [![`nemo_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/nemo_jp60.yml?label=nemo:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/nemo_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`onnx`](/packages/ml/onnx) | [![`onnx_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/onnx_jp46.yml?label=onnx:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/onnx_jp46.yml) [![`onnx_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/onnx_jp51.yml?label=onnx:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/onnx_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`onnxruntime:1.17`](/packages/ml/onnxruntime) |  |
| &nbsp;&nbsp;&nbsp; [`onnxruntime:1.17-builder`](/packages/ml/onnxruntime) |  |
| &nbsp;&nbsp;&nbsp; [`openai-triton:2.1.0`](/packages/ml/openai-triton) |  |
| &nbsp;&nbsp;&nbsp; [`openai-triton:2.1.0-builder`](/packages/ml/openai-triton) |  |
| &nbsp;&nbsp;&nbsp; [`openai-triton:3.0.0`](/packages/ml/openai-triton) |  |
| &nbsp;&nbsp;&nbsp; [`openai-triton:3.0.0-builder`](/packages/ml/openai-triton) |  |
| &nbsp;&nbsp;&nbsp; [`tensorboard`](/packages/tensorflow/tensorboard) |  |
| &nbsp;&nbsp;&nbsp; [`tensorflow2`](/packages/tensorflow) | [![`tensorflow2_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tensorflow2_jp46.yml?label=tensorflow2:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tensorflow2_jp46.yml) [![`tensorflow2_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tensorflow2_jp51.yml?label=tensorflow2:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tensorflow2_jp51.yml) [![`tensorflow2_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tensorflow2_jp60.yml?label=tensorflow2:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tensorflow2_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`tritonserver`](/packages/ml/tritonserver) | [![`tritonserver_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tritonserver_jp46.yml?label=tritonserver:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tritonserver_jp46.yml) [![`tritonserver_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tritonserver_jp51.yml?label=tritonserver:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tritonserver_jp51.yml) [![`tritonserver_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tritonserver_jp60.yml?label=tritonserver:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tritonserver_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`tvm`](/packages/ml/tvm) | [![`tvm_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tvm_jp51.yml?label=tvm:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tvm_jp51.yml) |
| <a id="multimedia">**`MULTIMEDIA`**</a> | |
| &nbsp;&nbsp;&nbsp; [`ffmpeg`](/packages/multimedia/ffmpeg) |  |
| &nbsp;&nbsp;&nbsp; [`gstreamer`](/packages/multimedia/gstreamer) | [![`gstreamer_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/gstreamer_jp46.yml?label=gstreamer:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/gstreamer_jp46.yml) [![`gstreamer_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/gstreamer_jp51.yml?label=gstreamer:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/gstreamer_jp51.yml) [![`gstreamer_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/gstreamer_jp60.yml?label=gstreamer:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/gstreamer_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`pyav`](/packages/multimedia/pyav) |  |
| <a id="nerf">**`NERF`**</a> | |
| &nbsp;&nbsp;&nbsp; [`fruitnerf:1.0`](/packages/nerf/fruitnerf) |  |
| &nbsp;&nbsp;&nbsp; [`gsplat:1.3.0`](/packages/nerf/gsplat) |  |
| &nbsp;&nbsp;&nbsp; [`gsplat:1.3.0-builder`](/packages/nerf/gsplat) |  |
| &nbsp;&nbsp;&nbsp; [`hloc:1.4`](/packages/nerf/hloc) |  |
| &nbsp;&nbsp;&nbsp; [`hloc:1.4-builder`](/packages/nerf/hloc) |  |
| &nbsp;&nbsp;&nbsp; [`manifold:2.5.1`](/packages/nerf/manifold) |  |
| &nbsp;&nbsp;&nbsp; [`manifold:2.5.1-builder`](/packages/nerf/manifold) |  |
| &nbsp;&nbsp;&nbsp; [`meshlab:MeshLab-2023.12`](/packages/nerf/meshlab) |  |
| &nbsp;&nbsp;&nbsp; [`meshlab:MeshLab-2023.12-builder`](/packages/nerf/meshlab) |  |
| &nbsp;&nbsp;&nbsp; [`nerfacc:0.5.3`](/packages/nerf/nerfacc) |  |
| &nbsp;&nbsp;&nbsp; [`nerfacc:0.5.3-builder`](/packages/nerf/nerfacc) |  |
| &nbsp;&nbsp;&nbsp; [`nerfstudio:0.3.2`](/packages/nerf/nerfstudio) |  |
| &nbsp;&nbsp;&nbsp; [`nerfstudio:0.3.2-builder`](/packages/nerf/nerfstudio) |  |
| &nbsp;&nbsp;&nbsp; [`nerfstudio:1.1.4`](/packages/nerf/nerfstudio) |  |
| &nbsp;&nbsp;&nbsp; [`nerfstudio:1.1.4-builder`](/packages/nerf/nerfstudio) |  |
| &nbsp;&nbsp;&nbsp; [`pixsfm:1.0`](/packages/nerf/pixsfm) |  |
| &nbsp;&nbsp;&nbsp; [`pixsfm:1.0-builder`](/packages/nerf/pixsfm) |  |
| &nbsp;&nbsp;&nbsp; [`polyscope:2.3.0`](/packages/nerf/polyscope) |  |
| &nbsp;&nbsp;&nbsp; [`polyscope:2.3.0-builder`](/packages/nerf/polyscope) |  |
| &nbsp;&nbsp;&nbsp; [`pyceres:2.3`](/packages/nerf/pyceres) |  |
| &nbsp;&nbsp;&nbsp; [`pyceres:2.3-builder`](/packages/nerf/pyceres) |  |
| &nbsp;&nbsp;&nbsp; [`pycolmap:3.10`](/packages/nerf/pycolmap) |  |
| &nbsp;&nbsp;&nbsp; [`pycolmap:3.10-builder`](/packages/nerf/pycolmap) |  |
| &nbsp;&nbsp;&nbsp; [`pycolmap:3.8`](/packages/nerf/pycolmap) |  |
| &nbsp;&nbsp;&nbsp; [`pycolmap:3.8-builder`](/packages/nerf/pycolmap) |  |
| &nbsp;&nbsp;&nbsp; [`pymeshlab:2023.12.post1`](/packages/nerf/pymeshlab) |  |
| &nbsp;&nbsp;&nbsp; [`pymeshlab:2023.12.post1-builder`](/packages/nerf/pymeshlab) |  |
| &nbsp;&nbsp;&nbsp; [`tinycudann:1.7`](/packages/nerf/tinycudann) |  |
| &nbsp;&nbsp;&nbsp; [`tinycudann:1.7-builder`](/packages/nerf/tinycudann) |  |
| &nbsp;&nbsp;&nbsp; [`vhacdx:0.0.8.post1`](/packages/nerf/vhacdx) |  |
| &nbsp;&nbsp;&nbsp; [`vhacdx:0.0.8.post1-builder`](/packages/nerf/vhacdx) |  |
| <a id="other">**`OTHER`**</a> | |
| &nbsp;&nbsp;&nbsp; [`tensorrt:8.6`](/packages/tensorrt) | [![`tensorrt-86_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tensorrt-86_jp60.yml?label=tensorrt-86:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tensorrt-86_jp60.yml) |
| <a id="pytorch">**`PYTORCH`**</a> | |
| &nbsp;&nbsp;&nbsp; [`pytorch:2.1`](/packages/pytorch) | [![`pytorch-21_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/pytorch-21_jp51.yml?label=pytorch-21:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/pytorch-21_jp51.yml) [![`pytorch-21_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/pytorch-21_jp60.yml?label=pytorch-21:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/pytorch-21_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`pytorch:2.1-builder`](/packages/pytorch) |  |
| &nbsp;&nbsp;&nbsp; [`pytorch:2.2`](/packages/pytorch) |  |
| &nbsp;&nbsp;&nbsp; [`pytorch:2.2-builder`](/packages/pytorch) |  |
| &nbsp;&nbsp;&nbsp; [`pytorch:2.3`](/packages/pytorch) |  |
| &nbsp;&nbsp;&nbsp; [`pytorch:2.3-builder`](/packages/pytorch) |  |
| &nbsp;&nbsp;&nbsp; [`pytorch:2.4`](/packages/pytorch) |  |
| &nbsp;&nbsp;&nbsp; [`pytorch:2.4-builder`](/packages/pytorch) |  |
| &nbsp;&nbsp;&nbsp; [`torch2trt`](/packages/pytorch/torch2trt) | [![`torch2trt_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/torch2trt_jp46.yml?label=torch2trt:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/torch2trt_jp46.yml) [![`torch2trt_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/torch2trt_jp51.yml?label=torch2trt:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/torch2trt_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`torch_tensorrt`](/packages/pytorch/torch_tensorrt) | [![`torch_tensorrt_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/torch_tensorrt_jp46.yml?label=torch_tensorrt:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/torch_tensorrt_jp46.yml) [![`torch_tensorrt_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/torch_tensorrt_jp51.yml?label=torch_tensorrt:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/torch_tensorrt_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`torchaudio:2.1.0`](/packages/pytorch/torchaudio) |  |
| &nbsp;&nbsp;&nbsp; [`torchaudio:2.1.0-builder`](/packages/pytorch/torchaudio) |  |
| &nbsp;&nbsp;&nbsp; [`torchaudio:2.2.2`](/packages/pytorch/torchaudio) |  |
| &nbsp;&nbsp;&nbsp; [`torchaudio:2.2.2-builder`](/packages/pytorch/torchaudio) |  |
| &nbsp;&nbsp;&nbsp; [`torchaudio:2.3.0`](/packages/pytorch/torchaudio) |  |
| &nbsp;&nbsp;&nbsp; [`torchaudio:2.3.0-builder`](/packages/pytorch/torchaudio) |  |
| &nbsp;&nbsp;&nbsp; [`torchaudio:2.4.0`](/packages/pytorch/torchaudio) |  |
| &nbsp;&nbsp;&nbsp; [`torchaudio:2.4.0-builder`](/packages/pytorch/torchaudio) |  |
| &nbsp;&nbsp;&nbsp; [`torchvision:0.16.2`](/packages/pytorch/torchvision) |  |
| &nbsp;&nbsp;&nbsp; [`torchvision:0.16.2-builder`](/packages/pytorch/torchvision) |  |
| &nbsp;&nbsp;&nbsp; [`torchvision:0.17.2`](/packages/pytorch/torchvision) |  |
| &nbsp;&nbsp;&nbsp; [`torchvision:0.17.2-builder`](/packages/pytorch/torchvision) |  |
| &nbsp;&nbsp;&nbsp; [`torchvision:0.18.0`](/packages/pytorch/torchvision) |  |
| &nbsp;&nbsp;&nbsp; [`torchvision:0.18.0-builder`](/packages/pytorch/torchvision) |  |
| &nbsp;&nbsp;&nbsp; [`torchvision:0.19.0`](/packages/pytorch/torchvision) |  |
| &nbsp;&nbsp;&nbsp; [`torchvision:0.19.0-builder`](/packages/pytorch/torchvision) |  |
| <a id="rag">**`RAG`**</a> | |
| &nbsp;&nbsp;&nbsp; [`jetson-copilot`](/packages/rag/jetson-copilot) |  |
| &nbsp;&nbsp;&nbsp; [`langchain`](/packages/rag/langchain) | [![`langchain_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/langchain_jp51.yml?label=langchain:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/langchain_jp51.yml) [![`langchain_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/langchain_jp60.yml?label=langchain:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/langchain_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`langchain:samples`](/packages/rag/langchain) | [![`langchain-samples_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/langchain-samples_jp51.yml?label=langchain-samples:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/langchain-samples_jp51.yml) [![`langchain-samples_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/langchain-samples_jp60.yml?label=langchain-samples:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/langchain-samples_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`llama-index`](/packages/rag/llama-index) |  |
| &nbsp;&nbsp;&nbsp; [`llama-index:samples`](/packages/rag/llama-index) |  |
| <a id="rapids">**`RAPIDS`**</a> | |
| &nbsp;&nbsp;&nbsp; [`cudf:23.10.03`](/packages/ml/rapids/cudf) | [![`cudf-231003_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cudf-231003_jp60.yml?label=cudf-231003:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cudf-231003_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`cuml`](/packages/ml/rapids/cuml) | [![`cuml_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cuml_jp51.yml?label=cuml:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cuml_jp51.yml) |
| <a id="robots">**`ROBOTS`**</a> | |
| &nbsp;&nbsp;&nbsp;&nbsp; [`lerobot`](/packages/robots/lerobot) |  |
| <a id="ros">**`ROS`**</a> | |
| &nbsp;&nbsp;&nbsp; [`ros:foxy-desktop`](/packages/ros) | [![`ros-foxy-desktop_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-desktop_jp46.yml?label=ros-foxy-desktop:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-desktop_jp46.yml) [![`ros-foxy-desktop_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-desktop_jp51.yml?label=ros-foxy-desktop:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-desktop_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:foxy-foxglove`](/packages/ros) |  |
| &nbsp;&nbsp;&nbsp; [`ros:foxy-ros-base`](/packages/ros) | [![`ros-foxy-ros-base_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-ros-base_jp46.yml?label=ros-foxy-ros-base:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-ros-base_jp46.yml) [![`ros-foxy-ros-base_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-ros-base_jp51.yml?label=ros-foxy-ros-base:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-ros-base_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:foxy-ros-core`](/packages/ros) | [![`ros-foxy-ros-core_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-ros-core_jp46.yml?label=ros-foxy-ros-core:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-ros-core_jp46.yml) [![`ros-foxy-ros-core_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-ros-core_jp51.yml?label=ros-foxy-ros-core:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-ros-core_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:galactic-desktop`](/packages/ros) | [![`ros-galactic-desktop_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-desktop_jp46.yml?label=ros-galactic-desktop:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-desktop_jp46.yml) [![`ros-galactic-desktop_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-desktop_jp51.yml?label=ros-galactic-desktop:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-desktop_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:galactic-foxglove`](/packages/ros) |  |
| &nbsp;&nbsp;&nbsp; [`ros:galactic-ros-base`](/packages/ros) | [![`ros-galactic-ros-base_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-ros-base_jp46.yml?label=ros-galactic-ros-base:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-ros-base_jp46.yml) [![`ros-galactic-ros-base_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-ros-base_jp51.yml?label=ros-galactic-ros-base:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-ros-base_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:galactic-ros-core`](/packages/ros) | [![`ros-galactic-ros-core_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-ros-core_jp46.yml?label=ros-galactic-ros-core:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-ros-core_jp46.yml) [![`ros-galactic-ros-core_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-ros-core_jp51.yml?label=ros-galactic-ros-core:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-ros-core_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:humble-desktop`](/packages/ros) | [![`ros-humble-desktop_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-desktop_jp46.yml?label=ros-humble-desktop:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-desktop_jp46.yml) [![`ros-humble-desktop_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-desktop_jp51.yml?label=ros-humble-desktop:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-desktop_jp51.yml) [![`ros-humble-desktop_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-desktop_jp60.yml?label=ros-humble-desktop:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-desktop_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:humble-foxglove`](/packages/ros) |  |
| &nbsp;&nbsp;&nbsp; [`ros:humble-ros-base`](/packages/ros) | [![`ros-humble-ros-base_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-base_jp46.yml?label=ros-humble-ros-base:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-base_jp46.yml) [![`ros-humble-ros-base_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-base_jp51.yml?label=ros-humble-ros-base:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-base_jp51.yml) [![`ros-humble-ros-base_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-base_jp60.yml?label=ros-humble-ros-base:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-base_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:humble-ros-core`](/packages/ros) | [![`ros-humble-ros-core_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-core_jp46.yml?label=ros-humble-ros-core:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-core_jp46.yml) [![`ros-humble-ros-core_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-core_jp51.yml?label=ros-humble-ros-core:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-core_jp51.yml) [![`ros-humble-ros-core_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-core_jp60.yml?label=ros-humble-ros-core:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-core_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:iron-desktop`](/packages/ros) | [![`ros-iron-desktop_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-desktop_jp46.yml?label=ros-iron-desktop:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-desktop_jp46.yml) [![`ros-iron-desktop_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-desktop_jp51.yml?label=ros-iron-desktop:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-desktop_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:iron-foxglove`](/packages/ros) |  |
| &nbsp;&nbsp;&nbsp; [`ros:iron-ros-base`](/packages/ros) | [![`ros-iron-ros-base_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-ros-base_jp46.yml?label=ros-iron-ros-base:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-ros-base_jp46.yml) [![`ros-iron-ros-base_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-ros-base_jp51.yml?label=ros-iron-ros-base:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-ros-base_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:iron-ros-core`](/packages/ros) | [![`ros-iron-ros-core_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-ros-core_jp46.yml?label=ros-iron-ros-core:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-ros-core_jp46.yml) [![`ros-iron-ros-core_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-ros-core_jp51.yml?label=ros-iron-ros-core:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-ros-core_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:jazzy-desktop`](/packages/ros) |  |
| &nbsp;&nbsp;&nbsp; [`ros:jazzy-foxglove`](/packages/ros) |  |
| &nbsp;&nbsp;&nbsp; [`ros:jazzy-ros-base`](/packages/ros) |  |
| &nbsp;&nbsp;&nbsp; [`ros:jazzy-ros-core`](/packages/ros) |  |
| &nbsp;&nbsp;&nbsp; [`ros:noetic-desktop`](/packages/ros) | [![`ros-noetic-desktop_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-desktop_jp46.yml?label=ros-noetic-desktop:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-desktop_jp46.yml) [![`ros-noetic-desktop_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-desktop_jp51.yml?label=ros-noetic-desktop:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-desktop_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:noetic-ros-base`](/packages/ros) | [![`ros-noetic-ros-base_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-ros-base_jp46.yml?label=ros-noetic-ros-base:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-ros-base_jp46.yml) [![`ros-noetic-ros-base_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-ros-base_jp51.yml?label=ros-noetic-ros-base:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-ros-base_jp51.yml) |
| &nbsp;&nbsp;&nbsp; [`ros:noetic-ros-core`](/packages/ros) | [![`ros-noetic-ros-core_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-ros-core_jp46.yml?label=ros-noetic-ros-core:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-ros-core_jp46.yml) [![`ros-noetic-ros-core_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-ros-core_jp51.yml?label=ros-noetic-ros-core:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-ros-core_jp51.yml) |
| <a id="sim">**`SIM`**</a> | |
| &nbsp;&nbsp;&nbsp; [`mimicgen`](/packages/robots/sim/mimicgen) |  |
| &nbsp;&nbsp;&nbsp; [`robomimic`](/packages/robots/sim/robomimic) |  |
| &nbsp;&nbsp;&nbsp; [`robosuite`](/packages/robots/sim/robosuite) |  |
| <a id="smart-home">**`SMART-HOME`**</a> | |
| &nbsp;&nbsp;&nbsp; [`homeassistant-base`](/packages/smart-home/homeassistant-base) |  |
| &nbsp;&nbsp;&nbsp; [`homeassistant-core:2024.4.2`](/packages/smart-home/homeassistant-core) |  |
| &nbsp;&nbsp;&nbsp; [`homeassistant-core:latest`](/packages/smart-home/homeassistant-core) |  |
| <a id="transformer">**`TRANSFORMER`**</a> | |
| &nbsp;&nbsp;&nbsp; [`ctranslate2:4.2.0`](/packages/ml/ctranslate2) |  |
| &nbsp;&nbsp;&nbsp; [`ctranslate2:4.2.0-builder`](/packages/ml/ctranslate2) |  |
| &nbsp;&nbsp;&nbsp; [`ctranslate2:master`](/packages/ml/ctranslate2) |  |
| &nbsp;&nbsp;&nbsp; [`ctranslate2:master-builder`](/packages/ml/ctranslate2) |  |
| <a id="vectordb">**`VECTORDB`**</a> | |
| &nbsp;&nbsp;&nbsp; [`faiss:1.7.3`](/packages/vectordb/faiss) |  |
| &nbsp;&nbsp;&nbsp; [`faiss:1.7.3-builder`](/packages/vectordb/faiss) |  |
| &nbsp;&nbsp;&nbsp; [`faiss:1.7.4`](/packages/vectordb/faiss) |  |
| &nbsp;&nbsp;&nbsp; [`faiss:1.7.4-builder`](/packages/vectordb/faiss) |  |
| &nbsp;&nbsp;&nbsp; [`faiss_lite`](/packages/vectordb/faiss_lite) |  |
| &nbsp;&nbsp;&nbsp; [`nanodb`](/packages/vectordb/nanodb) | [![`nanodb_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/nanodb_jp51.yml?label=nanodb:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/nanodb_jp51.yml) [![`nanodb_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/nanodb_jp60.yml?label=nanodb:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/nanodb_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`raft`](/packages/ml/rapids/raft) |  |
| <a id="vit">**`VIT`**</a> | |
| &nbsp;&nbsp;&nbsp; [`clip_trt`](/packages/vit/clip_trt) |  |
| &nbsp;&nbsp;&nbsp; [`efficientvit`](/packages/vit/efficientvit) | [![`efficientvit_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/efficientvit_jp51.yml?label=efficientvit:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/efficientvit_jp51.yml) [![`efficientvit_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/efficientvit_jp60.yml?label=efficientvit:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/efficientvit_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`nanoowl`](/packages/vit/nanoowl) | [![`nanoowl_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/nanoowl_jp51.yml?label=nanoowl:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/nanoowl_jp51.yml) [![`nanoowl_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/nanoowl_jp60.yml?label=nanoowl:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/nanoowl_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`nanosam`](/packages/vit/nanosam) | [![`nanosam_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/nanosam_jp51.yml?label=nanosam:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/nanosam_jp51.yml) [![`nanosam_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/nanosam_jp60.yml?label=nanosam:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/nanosam_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`sam`](/packages/vit/sam) | [![`sam_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/sam_jp51.yml?label=sam:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/sam_jp51.yml) [![`sam_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/sam_jp60.yml?label=sam:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/sam_jp60.yml) |
| &nbsp;&nbsp;&nbsp; [`tam`](/packages/vit/tam) | [![`tam_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tam_jp51.yml?label=tam:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tam_jp51.yml) [![`tam_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tam_jp60.yml?label=tam:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tam_jp60.yml) |
| <a id="wyoming">**`WYOMING`**</a> | |
| &nbsp;&nbsp;&nbsp; [`wyoming-assist-microphone:latest`](/packages/smart-home/wyoming/assist-microphone) |  |
| &nbsp;&nbsp;&nbsp; [`wyoming-openwakeword:latest`](/packages/smart-home/wyoming/openwakeword) |  |
| &nbsp;&nbsp;&nbsp; [`wyoming-piper:master`](/packages/smart-home/wyoming/piper) |  |
| &nbsp;&nbsp;&nbsp; [`wyoming-whisper:latest`](/packages/smart-home/wyoming/wyoming-whisper) |  |
