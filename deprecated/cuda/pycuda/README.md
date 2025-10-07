# pycuda

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`pycuda`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) |
| &nbsp;&nbsp;&nbsp;Dependants | [`audiocraft`](/packages/speech/audiocraft) [`cv-cuda:0.15`](/packages/cv/cv-cuda) [`diffusion_policy`](/packages/diffusion/diffusion_policy) [`dli-nano-ai`](/packages/ml/dli/dli-nano-ai) [`efficientvit`](/packages/vit/efficientvit) [`jupyter_clickable_image_widget`](/packages/hw/jupyter_clickable_image_widget) [`jupyterlab:4.2.0`](/packages/code/jupyterlab) [`jupyterlab:4.2.0-myst`](/packages/code/jupyterlab) [`jupyterlab:latest`](/packages/code/jupyterlab) [`jupyterlab:latest-myst`](/packages/code/jupyterlab) [`l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) [`l4t-dynamo`](/packages/ml/l4t/l4t-dynamo) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`l4t-pytorch`](/packages/ml/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/ml/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/ml/l4t/l4t-tensorflow) [`l4t-text-generation`](/packages/ml/l4t/l4t-text-generation) [`langchain:samples`](/packages/rag/langchain) [`lerobot`](/packages/robots/lerobot) [`llama-index:samples`](/packages/rag/llama-index) [`openpi`](/packages/robots/openpi) [`pytorch:2.1-all`](/packages/pytorch) [`pytorch:2.2-all`](/packages/pytorch) [`pytorch:2.3-all`](/packages/pytorch) [`pytorch:2.3.1-all`](/packages/pytorch) [`pytorch:2.4-all`](/packages/pytorch) [`pytorch:2.5-all`](/packages/pytorch) [`pytorch:2.6-all`](/packages/pytorch) [`pytorch:2.7-all`](/packages/pytorch) [`pytorch:2.8-all`](/packages/pytorch) [`sam`](/packages/vit/sam) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`voice-pro`](/packages/speech/voice-pro) [`voicecraft`](/packages/speech/voicecraft) [`warp:1.7.0-all`](/packages/numeric/warp) [`warp:1.8.1-all`](/packages/numeric/warp) [`whisper`](/packages/speech/whisper) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/pycuda:r32.7.1`](https://hub.docker.com/r/dustynv/pycuda/tags) `(2023-12-06, 0.4GB)`<br>[`dustynv/pycuda:r35.2.1`](https://hub.docker.com/r/dustynv/pycuda/tags) `(2023-09-07, 5.0GB)`<br>[`dustynv/pycuda:r35.3.1`](https://hub.docker.com/r/dustynv/pycuda/tags) `(2023-08-29, 5.0GB)`<br>[`dustynv/pycuda:r35.4.1`](https://hub.docker.com/r/dustynv/pycuda/tags) `(2023-12-06, 5.0GB)`<br>[`dustynv/pycuda:r36.2.0`](https://hub.docker.com/r/dustynv/pycuda/tags) `(2023-12-06, 3.5GB)`<br>[`dustynv/pycuda:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/pycuda/tags) `(2025-03-03, 2.3GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/pycuda:r32.7.1`](https://hub.docker.com/r/dustynv/pycuda/tags) | `2023-12-06` | `arm64` | `0.4GB` |
| &nbsp;&nbsp;[`dustynv/pycuda:r35.2.1`](https://hub.docker.com/r/dustynv/pycuda/tags) | `2023-09-07` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/pycuda:r35.3.1`](https://hub.docker.com/r/dustynv/pycuda/tags) | `2023-08-29` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/pycuda:r35.4.1`](https://hub.docker.com/r/dustynv/pycuda/tags) | `2023-12-06` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/pycuda:r36.2.0`](https://hub.docker.com/r/dustynv/pycuda/tags) | `2023-12-06` | `arm64` | `3.5GB` |
| &nbsp;&nbsp;[`dustynv/pycuda:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/pycuda/tags) | `2025-03-03` | `arm64` | `2.3GB` |

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
jetson-containers run $(autotag pycuda)

# or explicitly specify one of the container images above
jetson-containers run dustynv/pycuda:r36.4.0-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/pycuda:r36.4.0-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag pycuda)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag pycuda) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build pycuda
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
