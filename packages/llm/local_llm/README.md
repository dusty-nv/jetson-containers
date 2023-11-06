# local_llm

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


<a href="https://www.youtube.com/watch?v=9ObzbbBTbcc"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_llava_clip.gif"></a>

* Optimized LLM inference engine with support for AWQ and MLC quantization, multimodal agents, and live ASR/TTS.

## Text Chat

As an initial example, first test the console-based chat demo from [`__main__.py`](__main__.py)

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag local_llm) \
  python3 -m local_llm --api=mlc --model=meta-llama/Llama-2-7b-chat-hf
```
> For Llama-2 models, see [here](packages/llm/transformers/README.md#llama2) to request your access token from HuggingFace

The model will automatically be quantized the first time it's loaded (in this case, with MLC W4A16 quantization)

### Command-Line Options

Some of the noteworthy command-line options can be found in [`utils/args.py`](utils/args.py)

|                        |                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------|
| **Models**             |                                                                                           |
| `--model`              | The repo/name of the original unquantized model from HuggingFace Hub (or local path)      |
| `--quant`              | Either the API-specific quantization method to use, or path to quantized model            |
| `--api`                | The LLM model backend to use (`mlc, awq, auto_gptq, hf`)                                  |
| **Prompts**            |                                                                                           |
| `--prompt`             | Run this query (can be text, or a path to .txt file, and can be specified multiple times) |
| `--system-prompt`      | Sets the system instruction used at the beginning of the chat sequence.                   |
| `--chat-template`      | Manually set the chat template (`llama-2`, `llava-1`, `vicuna-v1`)                        |
| **Generation**         |                                                                                           |
| `--max-new-tokens`     | The maximum number of output tokens to generate for each response (default: 128)          |
| `--min-new-tokens`     | The minimum number of output tokens to generate (default: -1, disabled)                   |
| `--do-sample`          | Use token sampling during output with `--temperature` and `--top-p` settings              |
| `--temperature`        | Controls randomness of output with `--do-sample` (lower is less random, default: 0.7)     |
| `--top-p`              | Controls determinism/diversity of output with `--do-sample` (default: 0.95)               |
| `--repetition-penalty` | Applies a penalty for repetitive outputs (default: 1.0, disabled)                         |

### Automated Prompts

During testing, you can specify prompts on the command-line that will run sequentially:

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag local_llm) \
  python3 -m local_llm --api=mlc --model=meta-llama/Llama-2-7b-chat-hf \
    --prompt 'hi, how are you?' \
    --prompt 'whats the square root of 900?' \
    --prompt 'whats the previous answer times 4?' \
    --prompt 'can I get a recipie for french onion soup?'
```

### Multimodal (Llava)

If you load the Llava-1.5 model instead, you can enter image files into the prompt, followed by questions about them:

```bash
./run.sh $(./autotag local_llm) \
  python3 -m local_llm --api=mlc --model=liuhaotian/llava-v1.5-13b \
    --prompt '/data/images/fruit.jpg' \
    --prompt 'what kind of fruits do you see?' \
    --prompt '/data/images/dogs.jpg' \
    --prompt 'what breed of dogs are in the image?' \
    --prompt '/data/images/path.jpg' \
    --prompt 'what does the sign say?'
```

You can also enter `reset` (or `--prompt 'reset'`) to reset the chat history between images or responses.

## Voice Chat

<a href="https://www.youtube.com/watch?v=wzLHAgDxMjQ"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_70b_yt.jpg" width="800px"></a>

To enable the web UI and ASR/TTS for live conversations ...


<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`local_llm`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`local_llm_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/local_llm_jp51.yml?label=local_llm:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/local_llm_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/rust) [`transformers`](/packages/llm/transformers) [`mlc`](/packages/llm/mlc) [`opencv`](/packages/opencv) [`gstreamer`](/packages/gstreamer) [`jetson-utils`](/packages/jetson-utils) [`riva-client:python`](/packages/audio/riva-client) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/local_llm:r35.2.1`](https://hub.docker.com/r/dustynv/local_llm/tags) `(2023-09-20, 9.3GB)`<br>[`dustynv/local_llm:r35.3.1`](https://hub.docker.com/r/dustynv/local_llm/tags) `(2023-09-20, 9.3GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/local_llm:r35.2.1`](https://hub.docker.com/r/dustynv/local_llm/tags) | `2023-09-20` | `arm64` | `9.3GB` |
| &nbsp;&nbsp;[`dustynv/local_llm:r35.3.1`](https://hub.docker.com/r/dustynv/local_llm/tags) | `2023-09-20` | `arm64` | `9.3GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag local_llm)

# or explicitly specify one of the container images above
./run.sh dustynv/local_llm:r35.3.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/local_llm:r35.3.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag local_llm)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag local_llm) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh local_llm
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
