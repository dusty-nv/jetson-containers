# genesis

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`genesis-world:0.2.2`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `genesis-world` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36.0.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`python`](/packages/build/python) [`rust`](/packages/build/rust) [`cmake`](/packages/build/cmake/cmake_pip) [`cudnn`](/packages/cuda/cudnn) [`numpy`](/packages/numeric/numpy) [`onnx`](/packages/ml/onnx) [`torch`](/packages/pytorch) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) [`ninja`](/packages/build/ninja) [`opengl`](/packages/multimedia/opengl) [`polyscope`](/packages/3d/3dvision/polyscope) [`pymeshlab`](/packages/3d/3dvision/pymeshlab) [`llvm:21`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`vtk`](/packages/sim/genesis/vtk) [`taichi`](/packages/sim/genesis/taichi) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`splashsurf`](/packages/sim/genesis/splashSurf) |
| &nbsp;&nbsp;&nbsp;Dependants | [`protomotions:2.5.0`](/packages/robots/protomotions) [`robogen`](/packages/sim/robogen) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/Genesis-Embodied-AI/Genesis |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/genesis:r36.4.0-cu128`](https://hub.docker.com/r/dustynv/genesis/tags) | `2025-02-18` | `arm64` | `9.6GB` |
| &nbsp;&nbsp;[`dustynv/genesis:r36.4.3-cu128-24.04`](https://hub.docker.com/r/dustynv/genesis/tags) | `2025-03-06` | `arm64` | `6.8GB` |

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
jetson-containers run $(autotag genesis)

# or explicitly specify one of the container images above
jetson-containers run dustynv/genesis:r36.4.3-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/genesis:r36.4.3-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag genesis)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag genesis) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build genesis
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
