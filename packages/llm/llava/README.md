# llava

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* LLaVa vision LLM from https://github.com/haotian-liu/LLaVA 
* Quantization is WIP :warning: (https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-gptq)

<img src="https://github.com/dusty-nv/jetson-containers/raw/master/data/images/hoover.jpg" width="400">

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
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/rust) [`bitsandbytes`](/packages/llm/bitsandbytes) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag llava)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host llava:35.4.1

```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag llava)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag llava) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh llava
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
