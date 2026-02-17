# nerfstudio

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`nerfstudio:1.1.7`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `nerfstudio` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`bitsandbytes`](/packages/llm/bitsandbytes) [`diffusers`](/packages/diffusion/diffusers) [`h5py`](/packages/build/h5py) [`ninja`](/packages/build/ninja) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opengl`](/packages/multimedia/opengl) [`polyscope`](/packages/3d/3dvision/polyscope) [`pymeshlab`](/packages/3d/3dvision/pymeshlab) [`tinycudann`](/packages/3d/3dvision/tinycudann) [`torchaudio`](/packages/pytorch/torchaudio) [`nerfacc`](/packages/3d/3dvision/nerfacc) [`torch`](/packages/pytorch) [`pyceres`](/packages/3d/3dvision/pyceres) [`pycolmap`](/packages/3d/3dvision/pycolmap) [`glomap`](/packages/3d/3dvision/glomap) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`opencv`](/packages/cv/opencv) [`hloc`](/packages/3d/3dvision/hloc) [`nerfview`](/packages/3d/gaussian_splatting/nerfview) [`gsplat`](/packages/3d/gaussian_splatting/gsplat) |
| &nbsp;&nbsp;&nbsp;Dependants | [`fruitnerf:1.0`](/packages/3d/nerf/fruitnerf) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/nerfstudio-project/nerfstudio |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/nerfstudio:r36.3.0`](https://hub.docker.com/r/dustynv/nerfstudio/tags) | `2024-09-02` | `arm64` | `7.7GB` |
| &nbsp;&nbsp;[`dustynv/nerfstudio:r36.4.0-cu128`](https://hub.docker.com/r/dustynv/nerfstudio/tags) | `2025-02-11` | `arm64` | `7.1GB` |

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
jetson-containers run $(autotag nerfstudio)

# or explicitly specify one of the container images above
jetson-containers run dustynv/nerfstudio:r36.4.0-cu128

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/nerfstudio:r36.4.0-cu128
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag nerfstudio)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag nerfstudio) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build nerfstudio
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
