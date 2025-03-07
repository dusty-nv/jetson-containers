# robopoint

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

Thanks to [Manuel Schweiger](https://github.com/mschweig) for contributing this container for [RoboPoint](https://robo-point.github.io/)!

### Start Server

This container will automatically run the CMD [`/opt/robopoint/start-server.sh`](start-server.sh) upon startup, which unless overridden with a different CMD, will first download the model specified by `ROBOPOINT_MODEL` environment variable (by default [`wentao-yuan/robopoint-v1-vicuna-v1.5-13b`](https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-13b)), and then load the model in the precision set by `ROBOPOINT_QUANTIZATION` (by default `int4`)

```bash
# running this container will download the model and start the server
jetson-containers run dustynv/robopoint:r36.4.0

# set ROBOPOINT_MODEL to specify HF model to download from @wentao-yuan (or local path)
# set ROBOPOINT_QUANTIZATION to int4/int8/fp16 (default is int4, with bitsandbytes --load_in_4bit)
jetson-containers run \
  -e ROBOPOINT_MODEL="wentao-yuan/robopoint-v1-vicuna-v1.5-13b" \
  -e ROBOPOINT_QUANTIZATION="int4" \
  dustynv/robopoint:r36.4.0
```

To override the default CMD and manually set flags to the model loader:

```bash
# for these flags, run 'python3 -m robopoint.serve.model_worker --help'
jetson-containers run \
  dustynv/robopoint:r36.4.0 \
    /opt/robopoint/start-server.sh --max-len 512
```

Extra flags to the startup script get appended to the `robopoint.serve.model_worker` command-line.

### Gradio UI

Launching the server above will also start a gradio web UI, reachable at `http://JETSON_IP:7860`

`<TODO SCREENSHOT>`

### Test Client

Although you can `import robopoint` into a Python script inside the container environment that loads & performs inference with the model directly, by default RoboPoint uses a client/server architecture similar in effect to LLM [`chat.completion`] microservices due to the model sizes and dependencies.  

The [`client.py`](client.py) uses REST requests to example processes a test image, and can be run inside or outside of container.  Since the heavy lifting is done inside the server, the client has lightweight dependencies (just install `pip install gradio_client` first if running this outside of container)

```bash
# mount in the examples so they can be edited from outside container
jetson-containers run \
  -v $(jetson-containers root)/packages/robots/robopoint:/mnt \
  dustynv/robopoint:r36.4.0 \
    python3 /mnt/client.py
```

The performance is currently ~2 seconds/image on AGX Orin with int4 (bitsandbytes), which is currently fine for initial experimentation before migrating to more intensive VLM optimizations (for example in NanoLLM or SGLang), and also is appropriate for use of the REST API to save time during the frequent reloads of the clientside logic related to the robotics or simulator integration that typically occur under development.
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`robopoint`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.5`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`flash-attention`](/packages/llm/flash-attention) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/robopoint:r36.4.0`](https://hub.docker.com/r/dustynv/robopoint/tags) `(2025-03-07, 5.3GB)`<br>[`dustynv/robopoint:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/robopoint/tags) `(2025-03-07, 4.7GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/robopoint:r36.4.0`](https://hub.docker.com/r/dustynv/robopoint/tags) | `2025-03-07` | `arm64` | `5.3GB` |
| &nbsp;&nbsp;[`dustynv/robopoint:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/robopoint/tags) | `2025-03-07` | `arm64` | `4.7GB` |

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
jetson-containers run $(autotag robopoint)

# or explicitly specify one of the container images above
jetson-containers run dustynv/robopoint:r36.4.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/robopoint:r36.4.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag robopoint)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag robopoint) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build robopoint
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
