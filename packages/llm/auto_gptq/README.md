# auto_gptq

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


AutoGPTQ from https://github.com/PanQiWei/AutoGPTQ (installed under `/opt/AutoGPTQ`)

### Inference Benchmark

Substitute the GPTQ model from [HuggingFace Hub](https://huggingface.co/models?search=gptq) (or model path) that you want to run:

```bash
./run.sh --workdir=/opt/AutoGPTQ/examples/benchmark/ $(./autotag auto_gptq) \
   python3 generation_speed.py --model_name_or_path TheBloke/LLaMA-7b-GPTQ --use_safetensors --max_new_tokens=128
```

If you get the error `Exllama kernel does not support query/key/value fusion with act-order`, try adding `--no_inject_fused_attention`

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`auto_gptq:0.7.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `auto_gptq` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch:2.2`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dependants | [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/auto_gptq:r35.2.1`](https://hub.docker.com/r/dustynv/auto_gptq/tags) | `2023-12-15` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/auto_gptq:r35.3.1`](https://hub.docker.com/r/dustynv/auto_gptq/tags) | `2023-12-11` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/auto_gptq:r35.4.1`](https://hub.docker.com/r/dustynv/auto_gptq/tags) | `2023-12-14` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/auto_gptq:r36.2.0`](https://hub.docker.com/r/dustynv/auto_gptq/tags) | `2023-12-15` | `arm64` | `7.7GB` |

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
jetson-containers run $(autotag auto_gptq)

# or explicitly specify one of the container images above
jetson-containers run dustynv/auto_gptq:r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/auto_gptq:r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag auto_gptq)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag auto_gptq) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build auto_gptq
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
