# nanoowl

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* NanoOWL from https://github.com/NVIDIA-AI-IOT/nanoowl/

### Run the basic usage example and copy the result to host

```
jetson-containers run --workdir /opt/nanoowl/examples \
  $(autotag nanoowl) \
    python3 owl_predict.py \
      --prompt="[an owl, a glove]" \
      --threshold=0.1 \
      --image_encoder_engine=../data/owl_image_encoder_patch32.engine
```

### Run the tree prediction example (live camera)

1. First, Ensure you have a USB webcam device connected to your Jetson.

2. Launch the demo

```
jetson-containers run --workdir /opt/nanoowl/examples/tree_demo \
  $(autotag nanoowl) \
    python3 tree_demo.py ../../data/owl_image_encoder_patch32.engine
```

3. Second, open your browser to `http://<ip address>:7860`

> You can use a PC (or any machine) to open a web browser as long as  can access the Jetson via the network

4. Type whatever prompt you like to see what works! Here are some examples

  - Example: `[a face [a nose, an eye, a mouth]]`
  - Example: `[a face (interested, yawning / bored)]`
  - Example: `(indoors, outdoors)`

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`nanoowl`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`tensorrt`](/packages/cuda/tensorrt) [`torch2trt`](/packages/pytorch/torch2trt) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`opengl`](/packages/multimedia/opengl) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`gstreamer`](/packages/multimedia/gstreamer) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/nanoowl:r35.2.1`](https://hub.docker.com/r/dustynv/nanoowl/tags) `(2023-12-14, 7.1GB)`<br>[`dustynv/nanoowl:r35.3.1`](https://hub.docker.com/r/dustynv/nanoowl/tags) `(2024-02-22, 7.1GB)`<br>[`dustynv/nanoowl:r35.4.1`](https://hub.docker.com/r/dustynv/nanoowl/tags) `(2024-05-11, 7.0GB)`<br>[`dustynv/nanoowl:r36.2.0`](https://hub.docker.com/r/dustynv/nanoowl/tags) `(2024-05-11, 8.3GB)`<br>[`dustynv/nanoowl:r36.3.0`](https://hub.docker.com/r/dustynv/nanoowl/tags) `(2024-06-25, 8.3GB)`<br>[`dustynv/nanoowl:r36.4.0`](https://hub.docker.com/r/dustynv/nanoowl/tags) `(2025-02-07, 9.5GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/nanoowl:r35.2.1`](https://hub.docker.com/r/dustynv/nanoowl/tags) | `2023-12-14` | `arm64` | `7.1GB` |
| &nbsp;&nbsp;[`dustynv/nanoowl:r35.3.1`](https://hub.docker.com/r/dustynv/nanoowl/tags) | `2024-02-22` | `arm64` | `7.1GB` |
| &nbsp;&nbsp;[`dustynv/nanoowl:r35.4.1`](https://hub.docker.com/r/dustynv/nanoowl/tags) | `2024-05-11` | `arm64` | `7.0GB` |
| &nbsp;&nbsp;[`dustynv/nanoowl:r36.2.0`](https://hub.docker.com/r/dustynv/nanoowl/tags) | `2024-05-11` | `arm64` | `8.3GB` |
| &nbsp;&nbsp;[`dustynv/nanoowl:r36.3.0`](https://hub.docker.com/r/dustynv/nanoowl/tags) | `2024-06-25` | `arm64` | `8.3GB` |
| &nbsp;&nbsp;[`dustynv/nanoowl:r36.4.0`](https://hub.docker.com/r/dustynv/nanoowl/tags) | `2025-02-07` | `arm64` | `9.5GB` |

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
jetson-containers run $(autotag nanoowl)

# or explicitly specify one of the container images above
jetson-containers run dustynv/nanoowl:r36.4.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/nanoowl:r36.4.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag nanoowl)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag nanoowl) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build nanoowl
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
