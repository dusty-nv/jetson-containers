# llava

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* LLaVa vision LLM from https://github.com/haotian-liu/LLaVA 
* See [jetson-ai-lab.com/tutorial_llava.html](https://www.jetson-ai-lab.com/tutorial_llava.html) to use quantized with text-generation-webui

<img src="https://github.com/dusty-nv/jetson-containers/raw/master/data/images/hoover.jpg" width="400">

These are with `llava-llama-2` models, however `llava-1.5` is now the latest and can be use instead.

### llava-llama-2-7b-chat

This is a LoRA applied to the original llama-2-7b-chat model, hence you need to [request access](https://huggingface.co/meta-llama) and provide your [HuggingFace token](https://huggingface.co/docs/hub/security-tokens) (or use [`SaffalPoosh/llava-llama-2-7B-merged`](https://huggingface.co/SaffalPoosh/llava-llama-2-7B-merged) instead)

```bash
./run.sh --env HUGGING_FACE_HUB_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag llava) \
  python3 -m llava.serve.cli \
    --model-path liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview \
    --model-base meta-llama/Llama-2-7b-chat-hf \
    --image-file /data/images/hoover.jpg
```

```
USER: what does the road sign say?
ASSISTANT: The road sign says "Hoover Dam."

USER: how far away is the exit?
ASSISTANT: The exit is 1 mile away.

USER: what is the environment like?
ASSISTANT: The environment is desert-like, with a rocky landscape and a dirt road leading to the exit.
```

### llava-llama-2-13b-chat

```bash
./run.sh $(./autotag llava) \
  python3 -m llava.serve.cli \
    --model-path liuhaotian/llava-llama-2-13b-chat-lightning-preview \
    --image-file /data/images/hoover.jpg
```

```
USER: what does the text in the road sign say?
ASSISTANT: The text in the road sign says "Hoover Dam Exit 2 Mile."

USER: How far away is the exit?
ASSISTANT: The exit is two miles away from the current location.

USER: What kind of environment is it?
ASSISTANT: The environment is a desert setting, with a mountain in the background.
```

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`llava`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`triton`](/packages/ml/triton) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/llava:r35.2.1`](https://hub.docker.com/r/dustynv/llava/tags) `(2023-12-15, 6.3GB)`<br>[`dustynv/llava:r35.3.1`](https://hub.docker.com/r/dustynv/llava/tags) `(2023-12-12, 6.3GB)`<br>[`dustynv/llava:r35.4.1`](https://hub.docker.com/r/dustynv/llava/tags) `(2023-12-14, 6.3GB)`<br>[`dustynv/llava:r36.2.0`](https://hub.docker.com/r/dustynv/llava/tags) `(2023-12-18, 8.0GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/llava:r35.2.1`](https://hub.docker.com/r/dustynv/llava/tags) | `2023-12-15` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/llava:r35.3.1`](https://hub.docker.com/r/dustynv/llava/tags) | `2023-12-12` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/llava:r35.4.1`](https://hub.docker.com/r/dustynv/llava/tags) | `2023-12-14` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/llava:r36.2.0`](https://hub.docker.com/r/dustynv/llava/tags) | `2023-12-18` | `arm64` | `8.0GB` |

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
jetson-containers run $(autotag llava)

# or explicitly specify one of the container images above
jetson-containers run dustynv/llava:r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/llava:r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag llava)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag llava) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build llava
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
