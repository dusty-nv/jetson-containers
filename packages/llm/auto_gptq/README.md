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

| **`auto_gptq`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`auto_gptq_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/auto_gptq_jp51.yml?label=auto_gptq:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/auto_gptq_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`bitsandbytes`](/packages/llm/bitsandbytes) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/rust) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`text-generation-webui`](/packages/llm/text-generation-webui) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/auto_gptq:r35.2.1`](https://hub.docker.com/r/dustynv/auto_gptq/tags) `(2023-08-29, 5.9GB)`<br>[`dustynv/auto_gptq:r35.3.1`](https://hub.docker.com/r/dustynv/auto_gptq/tags) `(2023-08-29, 6.0GB)`<br>[`dustynv/auto_gptq:r35.4.1`](https://hub.docker.com/r/dustynv/auto_gptq/tags) `(2023-08-29, 5.9GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/auto_gptq:r35.2.1`](https://hub.docker.com/r/dustynv/auto_gptq/tags) | `2023-08-29` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/auto_gptq:r35.3.1`](https://hub.docker.com/r/dustynv/auto_gptq/tags) | `2023-08-29` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/auto_gptq:r35.4.1`](https://hub.docker.com/r/dustynv/auto_gptq/tags) | `2023-08-29` | `arm64` | `5.9GB` |

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
./run.sh $(./autotag auto_gptq)

# or explicitly specify one of the container images above
./run.sh dustynv/auto_gptq:r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/auto_gptq:r35.4.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag auto_gptq)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag auto_gptq) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh auto_gptq
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
