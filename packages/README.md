# Packages

> [`ATTENTION`](#attention) [`BUILD`](#build) [`CODE`](#code) [`CUDA`](#cuda) [`CV`](#cv) [`HW`](#hw) [`LLM`](#llm) [`ML`](#ml) [`MULTIMEDIA`](#multimedia) [`NET`](#net) [`PHYSICAL AI`](#physicalai) [`RAG`](#rag) [`SMART-HOME`](#smart-home) [`SPEECH`](#speech) [`VECTORDB`](#vectordb) [`VIT`](#vit) [`VLM`](#vlm)

---

## Attention

Attention mechanism implementations and optimizations.

| Package | Path |
|---------|------|
| `ParaAttention` | [`attention/ParaAttention`](/packages/attention/ParaAttention) |
| `block-sparse-attention` | [`attention/block-sparse-attention`](/packages/attention/block-sparse-attention) |
| `flash-attention` | [`attention/flash-attention`](/packages/attention/flash-attention) |
| `flashinfer` | [`attention/flashinfer`](/packages/attention/flashinfer) |
| `flexprefill` | [`attention/flexprefill`](/packages/attention/flexprefill) |
| `huggingface_kernels` | [`attention/huggingface_kernels`](/packages/attention/huggingface_kernels) |
| `jvp-flash-attention` | [`attention/jvp-flash-attention`](/packages/attention/jvp-flash-attention) |
| `log-linear-attention` | [`attention/log-linear-attention`](/packages/attention/log-linear-attention) |
| `radial-attention` | [`attention/radial-attention`](/packages/attention/radial-attention) |
| `sage-attention` | [`attention/sage-attention`](/packages/attention/sage-attention) |
| `sparge-attention` | [`attention/sparge-attention`](/packages/attention/sparge-attention) |
| `tilelang` | [`attention/tilelang`](/packages/attention/tilelang) |
| `xattention` | [`attention/xattention`](/packages/attention/xattention) |
| `xformers` | [`attention/xformers`](/packages/attention/xformers) |

---

## Build

Build tools, compilers, and development dependencies.

| Package | Path |
|---------|------|
| `bazel` | [`build/bazel`](/packages/build/bazel) |
| `build-essential` | [`build/build-essential`](/packages/build/build-essential) |
| `cmake:apt` | [`build/cmake/cmake_apt`](/packages/build/cmake/cmake_apt) |
| `cmake:pip` | [`build/cmake/cmake_pip`](/packages/build/cmake/cmake_pip) |
| `docker` | [`build/docker`](/packages/build/docker) |
| `go` | [`build/go`](/packages/build/go) |
| `h5py` | [`build/h5py`](/packages/build/h5py) |
| `helm` | [`build/helm`](/packages/build/helm) |
| `llvm` | [`build/llvm`](/packages/build/llvm) |
| `ninja` | [`build/ninja`](/packages/build/ninja) |
| `nodejs` | [`build/nodejs`](/packages/build/nodejs) |
| `protobuf:apt` | [`build/protobuf/protobuf_apt`](/packages/build/protobuf/protobuf_apt) |
| `protobuf:cpp` | [`build/protobuf/protobuf_cpp`](/packages/build/protobuf/protobuf_cpp) |
| `pybind11` | [`build/pybind11`](/packages/build/pybind11) |
| `python` | [`build/python`](/packages/build/python) |
| `rust` | [`build/rust`](/packages/build/rust) |

---

## Code

Development environments and IDEs.

| Package | Path |
|---------|------|
| `jupyterlab` | [`code/jupyterlab`](/packages/code/jupyterlab) |
| `openhands` | [`code/openhands`](/packages/code/openhands) |
| `vscode` | [`code/vscode`](/packages/code/vscode) |

---

## CUDA

CUDA toolkit, libraries, and Python bindings.

| Package | Path |
|---------|------|
| `cuda` | [`cuda/cuda`](/packages/cuda/cuda) |
| `cuda-python` | [`cuda/cuda-python`](/packages/cuda/cuda-python) |
| `cudastack` | [`cuda/cudastack`](/packages/cuda/cudastack) |
| &nbsp;&nbsp; `cudnn_frontend` | [`cuda/cudastack/cudnn_frontend`](/packages/cuda/cudastack/cudnn_frontend) |
| `cutlass` | [`cuda/cutlass`](/packages/cuda/cutlass) |
| `pycuda` | [`cuda/pycuda`](/packages/cuda/pycuda) |

---

## CV

Computer vision, 3D reconstruction, diffusion models, and related packages.

### cv/3d/3dobjects

| Package | Path |
|---------|------|
| `partpacker` | [`cv/3d/3dobjects/partpacker`](/packages/cv/3d/3dobjects/partpacker) |
| `sparc3d` | [`cv/3d/3dobjects/sparc3d`](/packages/cv/3d/3dobjects/sparc3d) |

### cv/3d/3dvision

| Package | Path |
|---------|------|
| `glomap` | [`cv/3d/3dvision/glomap`](/packages/cv/3d/3dvision/glomap) |
| `hloc` | [`cv/3d/3dvision/hloc`](/packages/cv/3d/3dvision/hloc) |
| `kaolin` | [`cv/3d/3dvision/kaolin`](/packages/cv/3d/3dvision/kaolin) |
| `meshlab` | [`cv/3d/3dvision/meshlab`](/packages/cv/3d/3dvision/meshlab) |
| `nerfacc` | [`cv/3d/3dvision/nerfacc`](/packages/cv/3d/3dvision/nerfacc) |
| `open3d` | [`cv/3d/3dvision/open3d`](/packages/cv/3d/3dvision/open3d) |
| `pixsfm` | [`cv/3d/3dvision/pixsfm`](/packages/cv/3d/3dvision/pixsfm) |
| `polyscope` | [`cv/3d/3dvision/polyscope`](/packages/cv/3d/3dvision/polyscope) |
| `pyceres` | [`cv/3d/3dvision/pyceres`](/packages/cv/3d/3dvision/pyceres) |
| `pycolmap` | [`cv/3d/3dvision/pycolmap`](/packages/cv/3d/3dvision/pycolmap) |
| `pymeshlab` | [`cv/3d/3dvision/pymeshlab`](/packages/cv/3d/3dvision/pymeshlab) |
| `tinycudann` | [`cv/3d/3dvision/tinycudann`](/packages/cv/3d/3dvision/tinycudann) |
| `usdcore` | [`cv/3d/3dvision/usdcore`](/packages/cv/3d/3dvision/usdcore) |

### cv/3d/gaussian_splatting

| Package | Path |
|---------|------|
| `3dgrut` | [`cv/3d/gaussian_splatting/3dgrut`](/packages/cv/3d/gaussian_splatting/3dgrut) |
| `4k4d` | [`cv/3d/gaussian_splatting/4k4d`](/packages/cv/3d/gaussian_splatting/4k4d) |
| `easyvolcap` | [`cv/3d/gaussian_splatting/easyvolcap`](/packages/cv/3d/gaussian_splatting/easyvolcap) |
| `fast_gauss` | [`cv/3d/gaussian_splatting/fast_gauss`](/packages/cv/3d/gaussian_splatting/fast_gauss) |
| `gsplat` | [`cv/3d/gaussian_splatting/gsplat`](/packages/cv/3d/gaussian_splatting/gsplat) |
| `nerfview` | [`cv/3d/gaussian_splatting/nerfview`](/packages/cv/3d/gaussian_splatting/nerfview) |

### cv/3d/nerf

| Package | Path |
|---------|------|
| `fruitnerf` | [`cv/3d/nerf/fruitnerf`](/packages/cv/3d/nerf/fruitnerf) |
| `nerfstudio` | [`cv/3d/nerf/nerfstudio`](/packages/cv/3d/nerf/nerfstudio) |

### cv/diffusion

| Package | Path |
|---------|------|
| `3d_diffusion_policy` | [`cv/diffusion/3d_diffusion_policy`](/packages/cv/diffusion/3d_diffusion_policy) |
| `ai-toolkit` | [`cv/diffusion/ai-toolkit`](/packages/cv/diffusion/ai-toolkit) |
| `cache_edit` | [`cv/diffusion/cache_edit`](/packages/cv/diffusion/cache_edit) |
| `comfyui` | [`cv/diffusion/comfyui`](/packages/cv/diffusion/comfyui) |
| `diffusers` | [`cv/diffusion/diffusers`](/packages/cv/diffusion/diffusers) |
| `diffusion_cpp` | [`cv/diffusion/diffusion_cpp`](/packages/cv/diffusion/diffusion_cpp) |
| `diffusion_policy` | [`cv/diffusion/diffusion_policy`](/packages/cv/diffusion/diffusion_policy) |
| `fastgen` | [`cv/diffusion/fastgen`](/packages/cv/diffusion/fastgen) |
| `framepack` | [`cv/diffusion/framepack`](/packages/cv/diffusion/framepack) |
| `sdnext` | [`cv/diffusion/sdnext`](/packages/cv/diffusion/sdnext) |
| `self-forcing` | [`cv/diffusion/self-forcing`](/packages/cv/diffusion/self-forcing) |
| `stable-diffusion-webui` | [`cv/diffusion/stable-diffusion-webui`](/packages/cv/diffusion/stable-diffusion-webui) |

### cv (other)

| Package | Path |
|---------|------|
| `cv-cuda` | [`cv/cv-cuda`](/packages/cv/cv-cuda) |
| `deepstream` | [`cv/deepstream`](/packages/cv/deepstream) |
| `holoscan` | [`cv/holoscan`](/packages/cv/holoscan) |
| `jetson-inference` | [`cv/jetson-inference`](/packages/cv/jetson-inference) |
| `opencv` | [`cv/opencv`](/packages/cv/opencv) |
| `vpi` | [`cv/vpi`](/packages/cv/vpi) |

---

## HW

Hardware interfaces and device drivers.

| Package | Path |
|---------|------|
| `canable` | [`hw/canable`](/packages/hw/canable) |
| `canbus` | [`hw/canbus`](/packages/hw/canbus) |
| `cangaroo` | [`hw/cangaroo`](/packages/hw/cangaroo) |
| `jetcam` | [`hw/jetcam`](/packages/hw/jetcam) |
| `jupyter_clickable_image_widget` | [`hw/jupyter_clickable_image_widget`](/packages/hw/jupyter_clickable_image_widget) |
| `librealsense` | [`hw/librealsense`](/packages/hw/librealsense) |
| `oled` | [`hw/oled`](/packages/hw/oled) |
| `pl2303` | [`hw/pl2303`](/packages/hw/pl2303) |
| `zed` | [`hw/zed`](/packages/hw/zed) |

---

## LLM

Large language model inference, training, quantization, and serving frameworks.

| Package | Path |
|---------|------|
| `awq` | [`llm/awq`](/packages/llm/awq) |
| `bitsandbytes` | [`llm/bitsandbytes`](/packages/llm/bitsandbytes) |
| `deepspeed` | [`llm/deepspeed`](/packages/llm/deepspeed) |
| `dynamo` | [`llm/dynamo/dynamo`](/packages/llm/dynamo/dynamo) |
| &nbsp;&nbsp; `kai-scheduler` | [`llm/dynamo/kai-scheduler`](/packages/llm/dynamo/kai-scheduler) |
| &nbsp;&nbsp; `mooncake` | [`llm/dynamo/mooncake`](/packages/llm/dynamo/mooncake) |
| &nbsp;&nbsp; `nixl` | [`llm/dynamo/nixl`](/packages/llm/dynamo/nixl) |
| `exllama` | [`llm/exllama`](/packages/llm/exllama) |
| `gptqmodel` | [`llm/gptqmodel`](/packages/llm/gptqmodel) |
| `huggingface_hub` | [`llm/huggingface_hub`](/packages/llm/huggingface_hub) |
| `hymba` | [`llm/hymba`](/packages/llm/hymba) |
| `ktransformers` | [`llm/ktransformers`](/packages/llm/ktransformers) |
| `litellm` | [`llm/litellm`](/packages/llm/litellm) |
| `llama-factory` | [`llm/llama-factory`](/packages/llm/llama-factory) |
| `llama_cpp` | [`llm/llama_cpp`](/packages/llm/llama_cpp) |
| `llamaspeak` | [`llm/llamaspeak`](/packages/llm/llamaspeak) |
| `lobe_chat` | [`llm/lobe_chat`](/packages/llm/lobe_chat) |
| `local_llm` | [`llm/local_llm`](/packages/llm/local_llm) |
| `minference` | [`llm/minference`](/packages/llm/minference) |
| `mistral_common` | [`llm/mistral/mistral_common`](/packages/llm/mistral/mistral_common) |
| `mistral_rs` | [`llm/mistral/mistral_rs`](/packages/llm/mistral/mistral_rs) |
| `mlc` | [`llm/mlc`](/packages/llm/mlc) |
| `nano_llm` | [`llm/nano_llm`](/packages/llm/nano_llm) |
| `nemo` | [`llm/nemo`](/packages/llm/nemo) |
| `nvidia-modelopt` | [`llm/tensorrt_optimizer/nvidia-modelopt`](/packages/llm/tensorrt_optimizer/nvidia-modelopt) |
| `ollama` | [`llm/ollama`](/packages/llm/ollama) |
| `open-webui` | [`llm/open-webui`](/packages/llm/open-webui) |
| `openai` | [`llm/openai`](/packages/llm/openai) |
| `optimum` | [`llm/optimum`](/packages/llm/optimum) |
| `sglang` | [`llm/sglang`](/packages/llm/sglang) |
| `sudonim` | [`llm/sudonim`](/packages/llm/sudonim) |
| `tensorrt_llm` | [`llm/tensorrt_optimizer/tensorrt_llm`](/packages/llm/tensorrt_optimizer/tensorrt_llm) |
| `text-generation-inference` | [`llm/text-generation-inference`](/packages/llm/text-generation-inference) |
| `text-generation-webui` | [`llm/text-generation-webui`](/packages/llm/text-generation-webui) |
| `transformers` | [`llm/transformers`](/packages/llm/transformers) |
| `unsloth` | [`llm/unsloth`](/packages/llm/unsloth) |
| `vllm` | [`llm/vllm`](/packages/llm/vllm) |
| `xgrammar` | [`llm/xgrammar`](/packages/llm/xgrammar) |

---

## ML

Machine learning frameworks, PyTorch ecosystem, numeric libraries, and RAPIDS.

### ml/pytorch

| Package | Path |
|---------|------|
| `pytorch` | [`ml/pytorch`](/packages/ml/pytorch) |
| `apex` | [`ml/pytorch/apex`](/packages/ml/pytorch/apex) |
| `torch2trt` | [`ml/pytorch/torch2trt`](/packages/ml/pytorch/torch2trt) |
| `torch3d` | [`ml/pytorch/torch3d`](/packages/ml/pytorch/torch3d) |
| `torch_tensorrt` | [`ml/pytorch/torch_tensorrt`](/packages/ml/pytorch/torch_tensorrt) |
| `torchao` | [`ml/pytorch/torchao`](/packages/ml/pytorch/torchao) |
| `torchaudio` | [`ml/pytorch/torchaudio`](/packages/ml/pytorch/torchaudio) |
| `torchcodec` | [`ml/pytorch/torchcodec`](/packages/ml/pytorch/torchcodec) |
| `torchsaver` | [`ml/pytorch/torchsaver`](/packages/ml/pytorch/torchsaver) |
| `torchsde` | [`ml/pytorch/torchsde`](/packages/ml/pytorch/torchsde) |
| `torchtext` | [`ml/pytorch/torchtext`](/packages/ml/pytorch/torchtext) |
| `torchvision` | [`ml/pytorch/torchvision`](/packages/ml/pytorch/torchvision) |

### ml/tensorflow

| Package | Path |
|---------|------|
| `tensorflow` | [`ml/tensorflow`](/packages/ml/tensorflow) |
| `tensorboard` | [`ml/tensorflow/tensorboard`](/packages/ml/tensorflow/tensorboard) |

### ml/jax

| Package | Path |
|---------|------|
| `jax` | [`ml/jax`](/packages/ml/jax) |

### ml/mamba

| Package | Path |
|---------|------|
| `causalconv1d` | [`ml/mamba/causalconv1d`](/packages/ml/mamba/causalconv1d) |
| `cobra` | [`ml/mamba/cobra`](/packages/ml/mamba/cobra) |
| `dimba` | [`ml/mamba/dimba`](/packages/ml/mamba/dimba) |
| `mamba` | [`ml/mamba/mamba`](/packages/ml/mamba/mamba) |
| `mambavision` | [`ml/mamba/mambavision`](/packages/ml/mamba/mambavision) |
| `videomambasuite` | [`ml/mamba/videomambasuite`](/packages/ml/mamba/videomambasuite) |
| `zigma` | [`ml/mamba/zigma`](/packages/ml/mamba/zigma) |

### ml/numeric

| Package | Path |
|---------|------|
| `arrow` | [`ml/numeric/arrow`](/packages/ml/numeric/arrow) |
| `cupy` | [`ml/numeric/cupy`](/packages/ml/numeric/cupy) |
| `numba` | [`ml/numeric/numba`](/packages/ml/numeric/numba) |
| `numpy` | [`ml/numeric/numpy`](/packages/ml/numeric/numpy) |
| `warp` | [`ml/numeric/warp`](/packages/ml/numeric/warp) |

### ml/rapids

| Package | Path |
|---------|------|
| `cudf` | [`ml/rapids/cudf`](/packages/ml/rapids/cudf) |
| `cuml` | [`ml/rapids/cuml`](/packages/ml/rapids/cuml) |
| `raft` | [`ml/rapids/raft`](/packages/ml/rapids/raft) |

### ml/kans

| Package | Path |
|---------|------|
| `kat` | [`ml/kans/kat`](/packages/ml/kans/kat) |
| `pykan` | [`ml/kans/pykan`](/packages/ml/kans/pykan) |

### ml/xlstm

| Package | Path |
|---------|------|
| `mlstm_kernels` | [`ml/xlstm/mlstm_kernels`](/packages/ml/xlstm/mlstm_kernels) |
| `pltsm` | [`ml/xlstm/pltsm`](/packages/ml/xlstm/pltsm) |
| `xlstm` | [`ml/xlstm/xlstm`](/packages/ml/xlstm/xlstm) |

### ml/apache

| Package | Path |
|---------|------|
| `tvm` | [`ml/apache/tvm`](/packages/ml/apache/tvm) |
| `tvm-ffi` | [`ml/apache/tvm-ffi`](/packages/ml/apache/tvm-ffi) |

### ml (other)

| Package | Path |
|---------|------|
| `aim` | [`ml/aim`](/packages/ml/aim) |
| `ctranslate2` | [`ml/ctranslate2`](/packages/ml/ctranslate2) |
| `dli-nano-ai` | [`ml/dli/dli-nano-ai`](/packages/ml/dli/dli-nano-ai) |
| `onnx` | [`ml/onnx`](/packages/ml/onnx) |
| `onnxruntime` | [`ml/onnxruntime`](/packages/ml/onnxruntime) |
| `onnxruntime_genai` | [`ml/onnxruntime_genai`](/packages/ml/onnxruntime_genai) |
| `transformer-engine` | [`ml/transformer-engine`](/packages/ml/transformer-engine) |
| `triton` | [`ml/triton`](/packages/ml/triton) |
| `tritonserver` | [`ml/tritonserver`](/packages/ml/tritonserver) |

### ml/l4t

| Package | Path |
|---------|------|
| `l4t-diffusion` | [`ml/l4t/l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) |
| `l4t-dynamo` | [`ml/l4t/l4t-dynamo`](/packages/ml/l4t/l4t-dynamo) |
| `l4t-ml` | [`ml/l4t/l4t-ml`](/packages/ml/l4t/l4t-ml) |
| `l4t-pytorch` | [`ml/l4t/l4t-pytorch`](/packages/ml/l4t/l4t-pytorch) |
| `l4t-tensorflow` | [`ml/l4t/l4t-tensorflow`](/packages/ml/l4t/l4t-tensorflow) |
| `l4t-text-generation` | [`ml/l4t/l4t-text-generation`](/packages/ml/l4t/l4t-text-generation) |

---

## Multimedia

Media processing, video codecs, and streaming.

| Package | Path |
|---------|------|
| `decord` | [`multimedia/decord`](/packages/multimedia/decord) |
| `ffmpeg` | [`multimedia/ffmpeg`](/packages/multimedia/ffmpeg) |
| `gstreamer` | [`multimedia/gstreamer`](/packages/multimedia/gstreamer) |
| `jetson-utils` | [`multimedia/jetson-utils`](/packages/multimedia/jetson-utils) |
| `libcom` | [`multimedia/libcom`](/packages/multimedia/libcom) |
| `opengl` | [`multimedia/opengl`](/packages/multimedia/opengl) |
| `pyav` | [`multimedia/pyav`](/packages/multimedia/pyav) |
| `sound-utils` | [`multimedia/sound-utils`](/packages/multimedia/sound-utils) |
| `video-codec-sdk` | [`multimedia/video-codec-sdk`](/packages/multimedia/video-codec-sdk) |
| `vulkan` | [`multimedia/vulkan`](/packages/multimedia/vulkan) |

---

## Net

Networking, proxies, and self-hosted infrastructure.

| Package | Path |
|---------|------|
| `devpi` | [`net/devpi`](/packages/net/devpi) |
| `https-portal` | [`net/https-portal`](/packages/net/https-portal) |
| `mkdocs` | [`net/mkdocs`](/packages/net/mkdocs) |
| `nginx_proxy_manager` | [`net/nginx_proxy_manager`](/packages/net/nginx_proxy_manager) |
| `portainer` | [`net/portainer`](/packages/net/portainer) |
| `pypi.dev` | [`net/pypi.dev`](/packages/net/pypi.dev) |
| `tailscale` | [`net/tailscale`](/packages/net/tailscale) |
| `yacht` | [`net/yacht`](/packages/net/yacht) |

---

## PhysicalAI

Robotics, simulation, embodied AI, and vision-language-action models.

### physicalAI/sim

| Package | Path |
|---------|------|
| `genesis` | [`physicalAI/sim/genesis`](/packages/physicalAI/sim/genesis) |
| &nbsp;&nbsp; `quadrants` | [`physicalAI/sim/genesis/quadrants`](/packages/physicalAI/sim/genesis/quadrants) |
| &nbsp;&nbsp; `splashSurf` | [`physicalAI/sim/genesis/splashSurf`](/packages/physicalAI/sim/genesis/splashSurf) |
| &nbsp;&nbsp; `vtk` | [`physicalAI/sim/genesis/vtk`](/packages/physicalAI/sim/genesis/vtk) |
| `habitat-sim` | [`physicalAI/sim/habitat-sim`](/packages/physicalAI/sim/habitat-sim) |
| `isaac-sim` | [`physicalAI/sim/isaac-sim`](/packages/physicalAI/sim/isaac-sim) |
| `mimicgen` | [`physicalAI/sim/mimicgen`](/packages/physicalAI/sim/mimicgen) |
| `mujoco` | [`physicalAI/sim/mujoco`](/packages/physicalAI/sim/mujoco) |
| `newton` | [`physicalAI/sim/newton`](/packages/physicalAI/sim/newton) |
| `physx` | [`physicalAI/sim/physx`](/packages/physicalAI/sim/physx) |
| `robogen` | [`physicalAI/sim/robogen`](/packages/physicalAI/sim/robogen) |
| `robomimic` | [`physicalAI/sim/robomimic`](/packages/physicalAI/sim/robomimic) |
| `robosuite` | [`physicalAI/sim/robosuite`](/packages/physicalAI/sim/robosuite) |
| `sapien` | [`physicalAI/sim/sapien`](/packages/physicalAI/sim/sapien) |

### physicalAI/vla

| Package | Path |
|---------|------|
| `crossformer` | [`physicalAI/vla/crossformer`](/packages/physicalAI/vla/crossformer) |
| `isaac-gr00t` | [`physicalAI/vla/isaac-gr00t`](/packages/physicalAI/vla/isaac-gr00t) |
| `octo` | [`physicalAI/vla/octo`](/packages/physicalAI/vla/octo) |
| `openvla` | [`physicalAI/vla/openvla`](/packages/physicalAI/vla/openvla) |
| `robopoint` | [`physicalAI/vla/robopoint`](/packages/physicalAI/vla/robopoint) |

### physicalAI/cosmos

| Package | Path |
|---------|------|
| `cosmos-policy` | [`physicalAI/cosmos/cosmos-policy`](/packages/physicalAI/cosmos/cosmos-policy) |
| `cosmos-predict2` | [`physicalAI/cosmos/cosmos-predict2`](/packages/physicalAI/cosmos/cosmos-predict2) |
| `cosmos-reason1` | [`physicalAI/cosmos/cosmos-reason1`](/packages/physicalAI/cosmos/cosmos-reason1) |
| `cosmos-transfer1` | [`physicalAI/cosmos/cosmos-transfer1`](/packages/physicalAI/cosmos/cosmos-transfer1) |
| `cosmos_diffusion_renderer` | [`physicalAI/cosmos/cosmos_diffusion_renderer`](/packages/physicalAI/cosmos/cosmos_diffusion_renderer) |

### physicalAI (other)

| Package | Path |
|---------|------|
| `Isaac-GR00T` | [`physicalAI/Isaac-GR00T`](/packages/physicalAI/Isaac-GR00T) |
| `isaac-ros` | [`physicalAI/isaac-ros`](/packages/physicalAI/isaac-ros) |
| `lerobot` | [`physicalAI/lerobot`](/packages/physicalAI/lerobot) |
| `opendronemap` | [`physicalAI/opendronemap`](/packages/physicalAI/opendronemap) |
| `openpi` | [`physicalAI/openpi`](/packages/physicalAI/openpi) |
| `protomotions` | [`physicalAI/protomotions`](/packages/physicalAI/protomotions) |
| `ros` | [`physicalAI/ros`](/packages/physicalAI/ros) |

---

## RAG

Retrieval-augmented generation and knowledge tools.

| Package | Path |
|---------|------|
| `graphiti` | [`rag/graphiti`](/packages/rag/graphiti) |
| `jetson-copilot` | [`rag/jetson-copilot`](/packages/rag/jetson-copilot) |
| `langchain` | [`rag/langchain`](/packages/rag/langchain) |
| `llama-index` | [`rag/llama-index`](/packages/rag/llama-index) |
| `memvid` | [`rag/memvid`](/packages/rag/memvid) |
| `n8n` | [`rag/n8n`](/packages/rag/n8n) |

---

## Smart-Home

Home automation, voice assistants, and Wyoming protocol servers.

| Package | Path |
|---------|------|
| `homeassistant-base` | [`smart-home/homeassistant-base`](/packages/smart-home/homeassistant-base) |
| `homeassistant-core` | [`smart-home/homeassistant-core`](/packages/smart-home/homeassistant-core) |
| `ciso8601` | [`smart-home/dependencies/ciso8601`](/packages/smart-home/dependencies/ciso8601) |
| `psutil-home-assistant` | [`smart-home/dependencies/psutil-home-assistant`](/packages/smart-home/dependencies/psutil-home-assistant) |
| `wyoming-assist-microphone` | [`smart-home/wyoming/wyoming-assist-microphone`](/packages/smart-home/wyoming/wyoming-assist-microphone) |
| `wyoming-openwakeword` | [`smart-home/wyoming/wyoming-openwakeword`](/packages/smart-home/wyoming/wyoming-openwakeword) |
| `wyoming-piper` | [`smart-home/wyoming/wyoming-piper`](/packages/smart-home/wyoming/wyoming-piper) |
| `wyoming-whisper` | [`smart-home/wyoming/wyoming-whisper`](/packages/smart-home/wyoming/wyoming-whisper) |

---

## Speech

Speech recognition, text-to-speech, and audio processing.

| Package | Path |
|---------|------|
| `audiocraft` | [`speech/audiocraft`](/packages/speech/audiocraft) |
| `chatterbox-tts` | [`speech/chatterbox-tts`](/packages/speech/chatterbox-tts) |
| `espeak` | [`speech/espeak`](/packages/speech/espeak) |
| `faster-whisper` | [`speech/faster-whisper`](/packages/speech/faster-whisper) |
| `kokoro-tts-fastapi` | [`speech/kokoro-tts/kokoro-tts-fastapi`](/packages/speech/kokoro-tts/kokoro-tts-fastapi) |
| `kokoro-tts-hf` | [`speech/kokoro-tts/kokoro-tts-hf`](/packages/speech/kokoro-tts/kokoro-tts-hf) |
| `kokoro-tts-onnx` | [`speech/kokoro-tts/kokoro-tts-onnx`](/packages/speech/kokoro-tts/kokoro-tts-onnx) |
| `piper1-tts` | [`speech/piper1-tts`](/packages/speech/piper1-tts) |
| `riva-client` | [`speech/riva-client`](/packages/speech/riva-client) |
| `spark-tts` | [`speech/spark-tts`](/packages/speech/spark-tts) |
| `speaches` | [`speech/speaches`](/packages/speech/speaches) |
| `speech-dispatcher` | [`speech/speech-dispatcher`](/packages/speech/speech-dispatcher) |
| `voice-pro` | [`speech/voice-pro`](/packages/speech/voice-pro) |
| `voicecraft` | [`speech/voicecraft`](/packages/speech/voicecraft) |
| `whisper` | [`speech/whisper`](/packages/speech/whisper) |
| `whisper_trt` | [`speech/whisper_trt`](/packages/speech/whisper_trt) |
| `whisperx` | [`speech/whisperx`](/packages/speech/whisperx) |
| `xtts` | [`speech/xtts`](/packages/speech/xtts) |

---

## VectorDB

Vector databases and similarity search.

| Package | Path |
|---------|------|
| `faiss` | [`vectordb/faiss`](/packages/vectordb/faiss) |
| `faiss_lite` | [`vectordb/faiss_lite`](/packages/vectordb/faiss_lite) |
| `nanodb` | [`vectordb/nanodb`](/packages/vectordb/nanodb) |

---

## VIT

Vision transformer models and segmentation.

| Package | Path |
|---------|------|
| `clip_trt` | [`vit/clip_trt`](/packages/vit/clip_trt) |
| `efficientvit` | [`vit/efficientvit`](/packages/vit/efficientvit) |
| `nanoowl` | [`vit/nanoowl`](/packages/vit/nanoowl) |
| `nanosam` | [`vit/nanosam`](/packages/vit/nanosam) |
| `sam` | [`vit/sam`](/packages/vit/sam) |
| `sapiens` | [`vit/sapiens`](/packages/vit/sapiens) |
| `tam` | [`vit/tam`](/packages/vit/tam) |

---

## VLM

Vision-language models for multimodal understanding.

| Package | Path |
|---------|------|
| `gemma_vlm` | [`vlm/gemma_vlm`](/packages/vlm/gemma_vlm) |
| `lita` | [`vlm/lita`](/packages/vlm/lita) |
| `llama-vision` | [`vlm/llama-vision`](/packages/vlm/llama-vision) |
| `llava` | [`vlm/llava`](/packages/vlm/llava) |
| `minigpt4` | [`vlm/minigpt4`](/packages/vlm/minigpt4) |
| `prismatic` | [`vlm/prismatic`](/packages/vlm/prismatic) |
| `videollama` | [`vlm/videollama`](/packages/vlm/videollama) |
| `vila` | [`vlm/vila`](/packages/vlm/vila) |
| `vila-microservice` | [`vlm/vila-microservice`](/packages/vlm/vila-microservice) |
| `xtuner` | [`vlm/xtuner`](/packages/vlm/xtuner) |
