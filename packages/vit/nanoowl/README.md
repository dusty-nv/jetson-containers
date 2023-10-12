# nanoowl

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* NanoOWL from https://github.com/NVIDIA-AI-IOT/nanoowl/

### Run the basic usage example and copy the result to host

```
./run.sh $(./autotag nanoowl) /bin/bash -c " \
  cd /opt/nanoowl/examples/ && \
  python3 owl_predict.py \
    --prompt="[an owl, a glove]" \
    --threshold=0.1 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine "
```

### Run the tree prediction example (live camera)

1. First, Ensure you have a USB webcam device connected to your Jetson.

2. Launch the demo

```
./run.sh $(./autotag nanoowl) /bin/bash -c " \
  cd /opt/nanoowl/examples/tree_demo/ && \
  python3 tree_demo.py ../../data/owl_image_encoder_patch32.engine "
```

3. Second, open your browser to `http://<ip address>:7860`

> You can use a PC (or any machine) to open a web browser as long as  can access the Jetson via the network

4. Type whatever prompt you like to see what works! Here are some examples

  - Example: [a face [a nose, an eye, a mouth]]
  - Example: [a face (interested, yawning / bored)]
  - Example: (indoors, outdoors)
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`nanoowl`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`nanoowl_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/nanoowl_jp51.yml?label=nanoowl:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/nanoowl_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torch2trt`](/packages/pytorch/torch2trt) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/rust) [`bitsandbytes`](/packages/llm/bitsandbytes) [`auto_gptq`](/packages/llm/auto_gptq) [`transformers`](/packages/llm/transformers) [`nanosam`](/packages/vit/nanosam) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/nanoowl:r35.4.1`](https://hub.docker.com/r/dustynv/nanoowl/tags) `(2023-10-12, 7.3GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/nanoowl:r35.4.1`](https://hub.docker.com/r/dustynv/nanoowl/tags) | `2023-10-12` | `arm64` | `7.3GB` |

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
./run.sh $(./autotag nanoowl)

# or explicitly specify one of the container images above
./run.sh dustynv/nanoowl:r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/nanoowl:r35.4.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag nanoowl)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag nanoowl) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh nanoowl
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
