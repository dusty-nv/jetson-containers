# mlc

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


Container for [MLC LLM](https://github.com/mlc-ai/mlc-llm) project using Apache TVM Unity with CUDA, cuDNN, CUTLASS, FasterTransformer, and FlashAttention-2 kernels.

### Benchmarks

To quantize and benchmark a model, run the [`benchmark.sh`](benchmark.sh) script from the host (outside container)

```bash
HUGGINGFACE_TOKEN=hf_abc123def ./benchmark.sh meta-llama/Llama-2-7b-hf
```

This will run the quantization and benchmarking in the MLC container, and save the performance data to `jetson-containers/data/benchmarks/mlc.csv`.  If you are accessing a gated model, substitute your HuggingFace account's API key above.  Omitting the model will benchmark a default set of Llama models.  See [`benchmark.sh`](benchmark.sh) for various environment variables you can set.

```
AVERAGE OVER 3 RUNS, input=16, output=128
/data/models/mlc/0.1.0/Llama-2-7b-hf-q4f16_ft/params:  prefill_time 0.025 sec, prefill_rate 632.8 tokens/sec, decode_time 2.731 sec, decode_rate 46.9 tokens/sec
```

The prefill time is how long the model takes to process the input context before it can start generating output tokens.  The decode rate is the speed at which it generates output tokens.  These results are averaged over the number of prompts, minus the first warm-up.

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`mlc:0.1.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `mlc` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch:2.2`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`local_llm`](/packages/llm/local_llm) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm/tree/607dc5a) commit SHA [`607dc5a`](https://github.com/mlc-ai/mlc-llm/tree/607dc5a) |

| **`mlc:0.1.0-builder`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `mlc` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch:2.2`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm/tree/607dc5a) commit SHA [`607dc5a`](https://github.com/mlc-ai/mlc-llm/tree/607dc5a) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/mlc:0.1.1-r36.2.0`](https://hub.docker.com/r/dustynv/mlc/tags) | `2024-04-18` | `arm64` | `7.4GB` |
| &nbsp;&nbsp;[`dustynv/mlc:3feed05-builder-r36.2.0`](https://hub.docker.com/r/dustynv/mlc/tags) | `2024-02-16` | `arm64` | `10.8GB` |
| &nbsp;&nbsp;[`dustynv/mlc:3feed05-r36.2.0`](https://hub.docker.com/r/dustynv/mlc/tags) | `2024-02-16` | `arm64` | `9.6GB` |
| &nbsp;&nbsp;[`dustynv/mlc:51fb0f4-builder-r35.4.1`](https://hub.docker.com/r/dustynv/mlc/tags) | `2024-02-16` | `arm64` | `9.5GB` |
| &nbsp;&nbsp;[`dustynv/mlc:51fb0f4-builder-r36.2.0`](https://hub.docker.com/r/dustynv/mlc/tags) | `2024-02-16` | `arm64` | `10.6GB` |
| &nbsp;&nbsp;[`dustynv/mlc:5584cac-r36.2.0`](https://hub.docker.com/r/dustynv/mlc/tags) | `2024-02-22` | `arm64` | `9.6GB` |
| &nbsp;&nbsp;[`dustynv/mlc:607dc5a-r36.2.0`](https://hub.docker.com/r/dustynv/mlc/tags) | `2024-02-27` | `arm64` | `9.6GB` |
| &nbsp;&nbsp;[`dustynv/mlc:c30348a-r36.2.0`](https://hub.docker.com/r/dustynv/mlc/tags) | `2024-02-20` | `arm64` | `9.6GB` |
| &nbsp;&nbsp;[`dustynv/mlc:dev-r35.3.1`](https://hub.docker.com/r/dustynv/mlc/tags) | `2023-10-30` | `arm64` | `9.0GB` |
| &nbsp;&nbsp;[`dustynv/mlc:dev-r35.4.1`](https://hub.docker.com/r/dustynv/mlc/tags) | `2023-12-16` | `arm64` | `9.4GB` |
| &nbsp;&nbsp;[`dustynv/mlc:dev-r36.2.0`](https://hub.docker.com/r/dustynv/mlc/tags) | `2023-12-16` | `arm64` | `10.6GB` |
| &nbsp;&nbsp;[`dustynv/mlc:r35.2.1`](https://hub.docker.com/r/dustynv/mlc/tags) | `2023-12-16` | `arm64` | `9.4GB` |
| &nbsp;&nbsp;[`dustynv/mlc:r35.3.1`](https://hub.docker.com/r/dustynv/mlc/tags) | `2023-11-05` | `arm64` | `8.9GB` |
| &nbsp;&nbsp;[`dustynv/mlc:r35.4.1`](https://hub.docker.com/r/dustynv/mlc/tags) | `2024-01-27` | `arm64` | `9.4GB` |
| &nbsp;&nbsp;[`dustynv/mlc:r36.2.0`](https://hub.docker.com/r/dustynv/mlc/tags) | `2024-03-09` | `arm64` | `9.6GB` |

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
jetson-containers run $(autotag mlc)

# or explicitly specify one of the container images above
jetson-containers run dustynv/mlc:0.1.1-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/mlc:0.1.1-r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag mlc)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag mlc) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build mlc
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
