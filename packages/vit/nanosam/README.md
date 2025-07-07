# nanosam

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* NanoSAM from https://github.com/NVIDIA-AI-IOT/nanosam/

### Run the basic usage example and copy the result to host

```
./run.sh $(./autotag nanosam) /bin/bash -c " \
  cd /opt/nanosam && \
  python3 examples/basic_usage.py \  
    --image_encoder=data/resnet18_image_encoder.engine \
    --mask_decoder=data/mobile_sam_mask_decoder.engine && \
  mv data/basic_usage_out.jpg /data/ \
  "
```

### Benchmark

```
 ./run.sh $(./autotag nanosam) /bin/bash -c " \
   cd /opt/nanosam && \
   python3 benchmark.py --run 3 -s /data/nanosam.csv && \
   mv data/benchmark_last_image.jpg /data/ \
   "
 ```
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`nanosam`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`tensorrt`](/packages/cuda/tensorrt) [`torch2trt`](/packages/pytorch/torch2trt) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/nanosam:r35.2.1`](https://hub.docker.com/r/dustynv/nanosam/tags) `(2023-12-15, 6.2GB)`<br>[`dustynv/nanosam:r35.3.1`](https://hub.docker.com/r/dustynv/nanosam/tags) `(2023-12-14, 6.2GB)`<br>[`dustynv/nanosam:r35.4.1`](https://hub.docker.com/r/dustynv/nanosam/tags) `(2023-11-05, 6.2GB)`<br>[`dustynv/nanosam:r36.2.0`](https://hub.docker.com/r/dustynv/nanosam/tags) `(2023-12-15, 7.9GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/nanosam:r35.2.1`](https://hub.docker.com/r/dustynv/nanosam/tags) | `2023-12-15` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/nanosam:r35.3.1`](https://hub.docker.com/r/dustynv/nanosam/tags) | `2023-12-14` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/nanosam:r35.4.1`](https://hub.docker.com/r/dustynv/nanosam/tags) | `2023-11-05` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/nanosam:r36.2.0`](https://hub.docker.com/r/dustynv/nanosam/tags) | `2023-12-15` | `arm64` | `7.9GB` |

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
jetson-containers run $(autotag nanosam)

# or explicitly specify one of the container images above
jetson-containers run dustynv/nanosam:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/nanosam:r35.2.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag nanosam)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag nanosam) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build nanosam
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
