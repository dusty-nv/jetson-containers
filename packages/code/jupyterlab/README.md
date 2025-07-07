# jupyterlab

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

Running the `jupyerlab` container will automatically start a JupyterLab server in the background on port 8888, with default login password `nvidia`.  The JupyterLab server logs will be saved to `/data/logs/jupyter.log` if you need to inspect them (this location is automatically mounted under your `jetson-containers/data` directory)

To change the default settings, you can set the `$JUPYTER_ROOT`, `$JUPYTER_PORT`, `$JUPYTER_PASSWORD`, and `$JUPYTER_LOG` environment variables when starting the container like so:

```bash
jetson-containers run \
  --env JUPYTER_ROOT=/home/user \
  --env JUPYTER_PORT=8000 \
  --env JUPYTER_PASSWORD=password \
  --env JUPYTER_LOGS=/dev/null \
  $(autotag jupyterlab)
```

The [`/start_jupyter`](./start_jupyter) script is the default CMD that the container runs when it starts - however, if you don't want the JupyterLab server started by default, you can either add a different CMD in your own Dockerfile, or override it at startup:

```bash
# skip straight to the terminal instead of starting JupyterLab first
jetson-containers run /bin/bash
```

You can then still manually run the [`/start_jupyter`](./start_jupyter) script later when desired.

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`jupyterlab:latest`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `jupyterlab` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`cuda-python`](/packages/cuda/cuda-python) [`pycuda`](/packages/cuda/pycuda) [`rust`](/packages/build/rust) |
| &nbsp;&nbsp;&nbsp;Dependants | [`audiocraft`](/packages/speech/audiocraft) [`diffusion_policy`](/packages/diffusion/diffusion_policy) [`dli-nano-ai`](/packages/ml/dli/dli-nano-ai) [`efficientvit`](/packages/vit/efficientvit) [`jupyter_clickable_image_widget`](/packages/hw/jupyter_clickable_image_widget) [`jupyterlab:latest-myst`](/packages/code/jupyterlab) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`l4t-text-generation`](/packages/ml/l4t/l4t-text-generation) [`langchain:samples`](/packages/rag/langchain) [`lerobot`](/packages/robots/lerobot) [`llama-index:samples`](/packages/rag/llama-index) [`openpi`](/packages/robots/openpi) [`pytorch:2.1-all`](/packages/pytorch) [`pytorch:2.2-all`](/packages/pytorch) [`pytorch:2.3-all`](/packages/pytorch) [`pytorch:2.3.1-all`](/packages/pytorch) [`pytorch:2.4-all`](/packages/pytorch) [`pytorch:2.5-all`](/packages/pytorch) [`pytorch:2.6-all`](/packages/pytorch) [`pytorch:2.7-all`](/packages/pytorch) [`pytorch:2.8-all`](/packages/pytorch) [`sam`](/packages/vit/sam) [`tam`](/packages/vit/tam) [`voice-pro`](/packages/speech/voice-pro) [`voicecraft`](/packages/speech/voicecraft) [`warp:1.7.0-all`](/packages/numeric/warp) [`warp:1.8.1-all`](/packages/numeric/warp) [`whisper`](/packages/speech/whisper) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | will autostart Jupyter server on port 8888 unless container entry CMD is overridden |

| **`jupyterlab:latest-myst`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `jupyterlab:myst` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`cuda-python`](/packages/cuda/cuda-python) [`pycuda`](/packages/cuda/pycuda) [`rust`](/packages/build/rust) [`jupyterlab:latest`](/packages/code/jupyterlab) |
| &nbsp;&nbsp;&nbsp;Dependants | [`lerobot`](/packages/robots/lerobot) [`llama-index:samples`](/packages/rag/llama-index) [`openpi`](/packages/robots/openpi) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.myst`](Dockerfile.myst) |
| &nbsp;&nbsp;&nbsp;Notes | will autostart Jupyter server on port 8888 unless container entry CMD is overridden |

| **`jupyterlab:4.2.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`cuda-python`](/packages/cuda/cuda-python) [`pycuda`](/packages/cuda/pycuda) [`rust`](/packages/build/rust) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jupyterlab:4.2.0-myst`](/packages/code/jupyterlab) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | will autostart Jupyter server on port 8888 unless container entry CMD is overridden |

| **`jupyterlab:4.2.0-myst`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`cuda-python`](/packages/cuda/cuda-python) [`pycuda`](/packages/cuda/pycuda) [`rust`](/packages/build/rust) [`jupyterlab:4.2.0`](/packages/code/jupyterlab) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.myst`](Dockerfile.myst) |
| &nbsp;&nbsp;&nbsp;Notes | will autostart Jupyter server on port 8888 unless container entry CMD is overridden |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r32.7.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2024-03-07` | `arm64` | `0.7GB` |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r35.2.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2023-12-06` | `arm64` | `5.3GB` |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r35.3.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2024-03-07` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r35.4.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2023-10-07` | `arm64` | `5.3GB` |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r36.2.0`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2024-03-07` | `arm64` | `0.6GB` |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r36.4.0`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2025-03-10` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2025-03-03` | `arm64` | `5.1GB` |

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
jetson-containers run $(autotag jupyterlab)

# or explicitly specify one of the container images above
jetson-containers run dustynv/jupyterlab:r36.4.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/jupyterlab:r36.4.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag jupyterlab)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag jupyterlab) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build jupyterlab
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
