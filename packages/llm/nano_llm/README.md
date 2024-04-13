# nano_llm

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


> [!NOTE]  
> [`NanoLLM`](https://dusty-nv.github.io/NanoLLM) is a lightweight, optimized library for LLM inference and multimodal agents.
> For more info, see these resources:
> * Repo - [`github.com/dusty-nv/NanoLLM`](https://github.com/dusty-nv/NanoLLM)
> * Docs - [`dusty-nv.github.io/NanoLLM`](https://dusty-nv.github.io/NanoLLM)
> * Jetson AI Lab - [Live Llava](https://www.jetson-ai-lab.com/tutorial_live-llava.html), [NanoVLM](https://www.jetson-ai-lab.com/tutorial_nano-vlm.html), [SLM](https://www.jetson-ai-lab.com/tutorial_slm.html)
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`nano_llm`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch:2.2`](/packages/pytorch) [`cuda-python`](/packages/cuda/cuda-python) [`faiss`](/packages/vectordb/faiss) [`faiss_lite`](/packages/vectordb/faiss_lite) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`tensorrt`](/packages/tensorrt) [`torch2trt`](/packages/pytorch/torch2trt) [`nanodb`](/packages/vectordb/nanodb) [`mlc`](/packages/llm/mlc) [`riva-client:python`](/packages/audio/riva-client) [`opencv`](/packages/opencv) [`gstreamer`](/packages/gstreamer) [`jetson-inference`](/packages/jetson-inference) [`onnxruntime`](/packages/onnxruntime) [`torchaudio`](/packages/pytorch/torchaudio) [`xtts`](/packages/audio/xtts) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag nano_llm)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host nano_llm:36.2.0

```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag nano_llm)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag nano_llm) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh nano_llm
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
