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
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torch2trt`](/packages/pytorch/torch2trt) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/rust) [`bitsandbytes`](/packages/llm/bitsandbytes) [`auto_gptq`](/packages/llm/auto_gptq) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag nanosam)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host nanosam:35.2.1

```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag nanosam)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag nanosam) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh nanosam
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
