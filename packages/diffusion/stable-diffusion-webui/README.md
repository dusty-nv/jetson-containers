# stable-diffusion-webui

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/diffusion_webui.jpg">

* stable-diffusion-webui from https://github.com/AUTOMATIC1111/stable-diffusion-webui (found under `/opt/stable-diffusion-webui`)
* with TensorRT extension from https://github.com/AUTOMATIC1111/stable-diffusion-webui-tensorrt 
* see the tutorial at the [**Jetson Generative AI Lab**](https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/tutorial_diffusion.html)

This container has a default run command that will automatically start the webserver like this:

```bash
cd /opt/stable-diffusion-webui && python3 launch.py \
  --data=/data/models/stable-diffusion \
  --enable-insecure-extension-access \
  --xformers \
  --listen \
  --port=7860
```

After starting the container, you can navigate your browser to `http://$IP_ADDRESS:7860` (substitute the address or hostname of your device).  The server will automatically download the default model ([`stable-diffusion-1.5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)) during startup.

Other configuration arguments can be found at [AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings)

* `--medvram` (sacrifice some performance for low VRAM usage)
* `--lowvram` (sacrafice a lot of speed for very low VRAM usage)

See the [`stable-diffusion`](/packages/diffusion/stable-diffusion) container to run image generation from a script (`txt2img.py`) as opposed to the web UI. 

### Tips & Tricks

Negative prompts:  https://huggingface.co/spaces/stabilityai/stable-diffusion/discussions/7857

Stable Diffusion XL
  * https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
  * https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
  * https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
  * https://stable-diffusion-art.com/sdxl-model/
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`stable-diffusion-webui`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`stable-diffusion-webui_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/stable-diffusion-webui_jp51.yml?label=stable-diffusion-webui:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/stable-diffusion-webui_jp51.yml) [![`stable-diffusion-webui_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/stable-diffusion-webui_jp60.yml?label=stable-diffusion-webui:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/stable-diffusion-webui_jp60.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/rust) [`transformers`](/packages/llm/transformers) [`xformers`](/packages/llm/xformers) [`pycuda`](/packages/cuda/pycuda) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-diffusion`](/packages/l4t/l4t-diffusion) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/stable-diffusion-webui:r35.2.1`](https://hub.docker.com/r/dustynv/stable-diffusion-webui/tags) `(2023-12-05, 7.1GB)`<br>[`dustynv/stable-diffusion-webui:r35.3.1`](https://hub.docker.com/r/dustynv/stable-diffusion-webui/tags) `(2023-11-05, 7.1GB)`<br>[`dustynv/stable-diffusion-webui:r35.4.1`](https://hub.docker.com/r/dustynv/stable-diffusion-webui/tags) `(2023-11-05, 7.1GB)`<br>[`dustynv/stable-diffusion-webui:r36.2.0`](https://hub.docker.com/r/dustynv/stable-diffusion-webui/tags) `(2023-12-05, 8.8GB)` |
| &nbsp;&nbsp;&nbsp;Notes | disabled on JetPack 4 |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/stable-diffusion-webui:r35.2.1`](https://hub.docker.com/r/dustynv/stable-diffusion-webui/tags) | `2023-12-05` | `arm64` | `7.1GB` |
| &nbsp;&nbsp;[`dustynv/stable-diffusion-webui:r35.3.1`](https://hub.docker.com/r/dustynv/stable-diffusion-webui/tags) | `2023-11-05` | `arm64` | `7.1GB` |
| &nbsp;&nbsp;[`dustynv/stable-diffusion-webui:r35.4.1`](https://hub.docker.com/r/dustynv/stable-diffusion-webui/tags) | `2023-11-05` | `arm64` | `7.1GB` |
| &nbsp;&nbsp;[`dustynv/stable-diffusion-webui:r36.2.0`](https://hub.docker.com/r/dustynv/stable-diffusion-webui/tags) | `2023-12-05` | `arm64` | `8.8GB` |

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
./run.sh $(./autotag stable-diffusion-webui)

# or explicitly specify one of the container images above
./run.sh dustynv/stable-diffusion-webui:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/stable-diffusion-webui:r35.2.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag stable-diffusion-webui)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag stable-diffusion-webui) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh stable-diffusion-webui
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
