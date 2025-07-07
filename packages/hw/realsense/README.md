# realsense

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`realsense`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/realsense:r32.7.1`](https://hub.docker.com/r/dustynv/realsense/tags) `(2024-02-22, 0.8GB)`<br>[`dustynv/realsense:r35.2.1`](https://hub.docker.com/r/dustynv/realsense/tags) `(2023-08-29, 5.5GB)`<br>[`dustynv/realsense:r35.3.1`](https://hub.docker.com/r/dustynv/realsense/tags) `(2024-02-22, 5.4GB)`<br>[`dustynv/realsense:r35.4.1`](https://hub.docker.com/r/dustynv/realsense/tags) `(2023-10-07, 5.5GB)`<br>[`dustynv/realsense:r36.2.0`](https://hub.docker.com/r/dustynv/realsense/tags) `(2024-02-22, 4.0GB)` |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/realsense:r32.7.1`](https://hub.docker.com/r/dustynv/realsense/tags) | `2024-02-22` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/realsense:r35.2.1`](https://hub.docker.com/r/dustynv/realsense/tags) | `2023-08-29` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/realsense:r35.3.1`](https://hub.docker.com/r/dustynv/realsense/tags) | `2024-02-22` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/realsense:r35.4.1`](https://hub.docker.com/r/dustynv/realsense/tags) | `2023-10-07` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/realsense:r36.2.0`](https://hub.docker.com/r/dustynv/realsense/tags) | `2024-02-22` | `arm64` | `4.0GB` |

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
jetson-containers run $(autotag realsense)

# or explicitly specify one of the container images above
jetson-containers run dustynv/realsense:r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/realsense:r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag realsense)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag realsense) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build realsense
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
