# zed

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`zed`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`zed_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/zed_jp46.yml?label=zed:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/zed_jp46.yml) [![`zed_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/zed_jp51.yml?label=zed:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/zed_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/zed:r32.7.1`](https://hub.docker.com/r/dustynv/zed/tags) `(2023-09-07, 0.6GB)`<br>[`dustynv/zed:r35.2.1`](https://hub.docker.com/r/dustynv/zed/tags) `(2023-12-11, 5.2GB)`<br>[`dustynv/zed:r35.3.1`](https://hub.docker.com/r/dustynv/zed/tags) `(2023-08-29, 5.2GB)`<br>[`dustynv/zed:r35.4.1`](https://hub.docker.com/r/dustynv/zed/tags) `(2023-10-07, 5.1GB)` |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/stereolabs/zed-docker/blob/master/4.X/l4t/py-devel/Dockerfile |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/zed:r32.7.1`](https://hub.docker.com/r/dustynv/zed/tags) | `2023-09-07` | `arm64` | `0.6GB` |
| &nbsp;&nbsp;[`dustynv/zed:r35.2.1`](https://hub.docker.com/r/dustynv/zed/tags) | `2023-12-11` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/zed:r35.3.1`](https://hub.docker.com/r/dustynv/zed/tags) | `2023-08-29` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/zed:r35.4.1`](https://hub.docker.com/r/dustynv/zed/tags) | `2023-10-07` | `arm64` | `5.1GB` |

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
jetson-containers run $(autotag zed)

# or explicitly specify one of the container images above
jetson-containers run dustynv/zed:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/zed:r35.2.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag zed)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag zed) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build zed
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
