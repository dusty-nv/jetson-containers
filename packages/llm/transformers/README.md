# transformers

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


The HuggingFace [Transformers](https://huggingface.co/docs/transformers/index) library supports a wide variety of NLP and vision models with a convenient API, and is used by many of the other LLM packages.  There are a large number of models that it's compatible with on [HuggingFace Hub](https://huggingface.co/models).

### Text Generation Benchmark

Substitute the [text-generation model](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) that you want to run (it should be a CausalLM model like GPT, Llama, ect)

```bash
./run.sh $(./autotag transformers) \
   huggingface-benchmark.py --model=gpt2
```
> If the model repository is private or requires authentication, add `--env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN>`

By default, the performance is measured for generating 128 new output tokens (this can be set with `--tokens=N`)

The prompt can be changed with `--prompt='your prompt here'`

#### Precision / Quantization

Use the `--precision` argument to enable quantization (options are: `fp32` `fp16` `fp4` `int8`)

On JetPack 5, the [`bitsandbytes`](/packages/llm/bitsandbytes) package is included in the container to enable 4-bit/8-bit quantization through the Transformers API.  It's expected that 4-bit/8-bit quantization is slower through Transformers than FP16 (while consuming less memory).  Other libraries like [`exllama`](/packages/llm/exllama), [`awq`](/packages/llm/awq), and [`AutoGPTQ`](/packages/llm/auto-gptq) have custom CUDA kernels and more efficient quantized performance.  The default precision used is FP16.

#### Llama2

* First request access from https://ai.meta.com/llama/
* Then create a HuggingFace account, and request access to one of the Llama2 models there like https://huggingface.co/meta-llama/Llama-2-7b-hf (doing this will get you access to all the Llama2 models)
* Get a User Access Token from https://huggingface.co/settings/tokens

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag transformers) \
   huggingface-benchmark.py --model=meta-llama/Llama-2-7b-hf
```

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`transformers`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`transformers_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/transformers_jp51.yml?label=transformers:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/transformers_jp51.yml) [![`transformers_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/transformers_jp46.yml?label=transformers:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/transformers_jp46.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/rust) [`bitsandbytes`](/packages/llm/bitsandbytes) |
| &nbsp;&nbsp;&nbsp;Dependants | [`auto_gptq`](/packages/llm/auto_gptq) [`awq`](/packages/llm/awq) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`llava`](/packages/llm/llava) [`nemo`](/packages/nemo) [`optimum`](/packages/llm/optimum) [`stable-diffusion`](/packages/diffusion/stable-diffusion) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui`](/packages/llm/text-generation-webui) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/transformers:r32.7.1`](https://hub.docker.com/r/dustynv/transformers/tags) `(2023-08-21, 1.4GB)`<br>[`dustynv/transformers:r35.2.1`](https://hub.docker.com/r/dustynv/transformers/tags) `(2023-08-29, 5.8GB)`<br>[`dustynv/transformers:r35.3.1`](https://hub.docker.com/r/dustynv/transformers/tags) `(2023-08-29, 5.8GB)`<br>[`dustynv/transformers:r35.4.1`](https://hub.docker.com/r/dustynv/transformers/tags) `(2023-08-29, 5.8GB)` |
| &nbsp;&nbsp;&nbsp;Notes | bitsandbytes dependency added on JetPack5 for 4-bit/8-bit quantization |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/transformers:r32.7.1`](https://hub.docker.com/r/dustynv/transformers/tags) | `2023-08-21` | `arm64` | `1.4GB` |
| &nbsp;&nbsp;[`dustynv/transformers:r35.2.1`](https://hub.docker.com/r/dustynv/transformers/tags) | `2023-08-29` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/transformers:r35.3.1`](https://hub.docker.com/r/dustynv/transformers/tags) | `2023-08-29` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/transformers:r35.4.1`](https://hub.docker.com/r/dustynv/transformers/tags) | `2023-08-29` | `arm64` | `5.8GB` |

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
./run.sh $(./autotag transformers)

# or explicitly specify one of the container images above
./run.sh dustynv/transformers:r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/transformers:r35.4.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag transformers)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag transformers) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh transformers
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
