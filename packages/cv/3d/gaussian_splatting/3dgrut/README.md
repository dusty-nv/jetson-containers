# 3dgrut

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`3dgrut:2.0.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `3dgrut` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cupy`](/packages/numeric/cupy) [`cuda-python`](/packages/cuda/cuda-python) [`cudnn`](/packages/cuda/cudnn) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) [`pytorch3d`](/packages/pytorch/torch3d) [`tinycudann`](/packages/3d/3dvision/tinycudann) [`opengl`](/packages/multimedia/opengl) [`polyscope`](/packages/3d/3dvision/polyscope) [`kaolin`](/packages/3d/3dvision/kaolin) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/nv-tlabs/3dgrut |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag 3dgrut)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host 3dgrut:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag 3dgrut)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag 3dgrut) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build 3dgrut
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
