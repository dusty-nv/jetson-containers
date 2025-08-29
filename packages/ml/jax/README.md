# jax

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

Containers for JAX with CUDA support.
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`jax:0.4.38`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm:21`](/packages/build/llvm) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`jax:0.6.3`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `jax` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm:21`](/packages/build/llvm) |
| &nbsp;&nbsp;&nbsp;Dependants | [`crossformer`](/packages/vla/crossformer) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`octo`](/packages/vla/octo) [`openpi`](/packages/robots/openpi) [`warp:1.7.0-all`](/packages/numeric/warp) [`warp:1.7.0-jax`](/packages/numeric/warp) [`warp:1.8.1-all`](/packages/numeric/warp) [`warp:1.8.1-jax`](/packages/numeric/warp) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/jax:0.5.2-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/jax/tags) | `2025-03-03` | `arm64` | `3.6GB` |
| &nbsp;&nbsp;[`dustynv/jax:r36.3.0`](https://hub.docker.com/r/dustynv/jax/tags) | `2024-09-12` | `arm64` | `6.9GB` |
| &nbsp;&nbsp;[`dustynv/jax:r36.3.0-cu126`](https://hub.docker.com/r/dustynv/jax/tags) | `2024-09-13` | `arm64` | `6.9GB` |
| &nbsp;&nbsp;[`dustynv/jax:r36.4.0`](https://hub.docker.com/r/dustynv/jax/tags) | `2024-10-15` | `arm64` | `4.2GB` |

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
jetson-containers run $(autotag jax)

# or explicitly specify one of the container images above
jetson-containers run dustynv/jax:0.5.2-r36.4.0-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/jax:0.5.2-r36.4.0-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag jax)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag jax) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build jax
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
