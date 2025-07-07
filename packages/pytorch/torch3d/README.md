# torch3d

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`pytorch3d:0.7.8`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==36.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`cudnn`](/packages/cuda/cudnn) [`numpy`](/packages/numeric/numpy) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`pytorch3d:0.7.9`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `pytorch3d` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`cudnn`](/packages/cuda/cudnn) [`numpy`](/packages/numeric/numpy) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) |
| &nbsp;&nbsp;&nbsp;Dependants | [`3d_diffusion_policy`](/packages/diffusion/3d_diffusion_policy) [`3dgrut:2.0.0`](/packages/3d/gaussian_splatting/3dgrut) [`4k4d:0.0.0`](/packages/3d/gaussian_splatting/4k4d) [`comfyui`](/packages/diffusion/comfyui) [`diffusion_policy`](/packages/diffusion/diffusion_policy) [`easyvolcap:0.0.0`](/packages/3d/gaussian_splatting/easyvolcap) [`fast_gauss:1.0.0`](/packages/3d/gaussian_splatting/fast_gauss) [`isaac-gr00t`](/packages/vla/isaac-gr00t) [`l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) [`pytorch:2.1-all`](/packages/pytorch) [`pytorch:2.2-all`](/packages/pytorch) [`pytorch:2.3-all`](/packages/pytorch) [`pytorch:2.3.1-all`](/packages/pytorch) [`pytorch:2.4-all`](/packages/pytorch) [`pytorch:2.5-all`](/packages/pytorch) [`pytorch:2.6-all`](/packages/pytorch) [`pytorch:2.7-all`](/packages/pytorch) [`pytorch:2.8-all`](/packages/pytorch) [`self-forcing`](/packages/diffusion/self-forcing) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag torch3d)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host torch3d:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag torch3d)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag torch3d) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build torch3d
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
