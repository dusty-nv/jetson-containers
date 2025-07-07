# minigpt4

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* minigpt4.cpp from https://github.com/Maknee/minigpt4.cpp with CUDA enabled (found under `/opt/minigpt4.cpp`)
* original MiniGPT-4 models and project are from https://github.com/Vision-CAIR/MiniGPT-4

To start the MiniGPT4 container and webserver with the [recommended models](https://github.com/Maknee/minigpt4.cpp/tree/master#3-obtaining-the-model), run this command:

```bash
./run.sh $(./autotag minigpt4) /bin/bash -c 'cd /opt/minigpt4.cpp/minigpt4 && python3 webui.py \
  $(huggingface-downloader --type=dataset maknee/minigpt4-13b-ggml/minigpt4-13B-f16.bin) \
  $(huggingface-downloader --type=dataset maknee/ggml-vicuna-v0-quantized/ggml-vicuna-13B-v0-q5_k.bin)'
```

Then navigate your browser to `http://HOSTNAME:7860`

### Inference Benchmark

```
./run.sh --workdir=/opt/minigpt4.cpp/minigpt4/ $(./autotag minigpt4) /bin/bash -c \
  'python3 benchmark.py \
    $(huggingface-downloader --type=dataset maknee/minigpt4-13b-ggml/minigpt4-13B-f16.bin) \
    $(huggingface-downloader --type=dataset maknee/ggml-vicuna-v0-quantized/ggml-vicuna-13B-v0-q5_k.bin) \
    --prompt "What does the sign say?" --prompt "How far is the exit?" --prompt "What would happen next?" \
    --image /data/images/hoover.jpg \
    --run 3 \
    --save /data/minigpt4.csv'
```



<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`minigpt4`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/minigpt4:r35.2.1`](https://hub.docker.com/r/dustynv/minigpt4/tags) `(2023-12-11, 5.9GB)`<br>[`dustynv/minigpt4:r35.3.1`](https://hub.docker.com/r/dustynv/minigpt4/tags) `(2023-12-15, 5.9GB)`<br>[`dustynv/minigpt4:r35.4.1`](https://hub.docker.com/r/dustynv/minigpt4/tags) `(2023-12-14, 5.9GB)`<br>[`dustynv/minigpt4:r36.2.0`](https://hub.docker.com/r/dustynv/minigpt4/tags) `(2023-12-15, 7.6GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/minigpt4:r35.2.1`](https://hub.docker.com/r/dustynv/minigpt4/tags) | `2023-12-11` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/minigpt4:r35.3.1`](https://hub.docker.com/r/dustynv/minigpt4/tags) | `2023-12-15` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/minigpt4:r35.4.1`](https://hub.docker.com/r/dustynv/minigpt4/tags) | `2023-12-14` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/minigpt4:r36.2.0`](https://hub.docker.com/r/dustynv/minigpt4/tags) | `2023-12-15` | `arm64` | `7.6GB` |

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
jetson-containers run $(autotag minigpt4)

# or explicitly specify one of the container images above
jetson-containers run dustynv/minigpt4:r35.3.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/minigpt4:r35.3.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag minigpt4)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag minigpt4) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build minigpt4
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
