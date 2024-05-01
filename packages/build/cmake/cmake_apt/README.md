# cmake_apt

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`cmake:apt`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`cmake-apt_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cmake-apt_jp51.yml?label=cmake-apt:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cmake-apt_jp51.yml) [![`cmake-apt_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cmake-apt_jp46.yml?label=cmake-apt:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cmake-apt_jp46.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) |
| &nbsp;&nbsp;&nbsp;Dependants | [`ros:melodic-desktop`](/packages/ros) [`ros:melodic-ros-base`](/packages/ros) [`ros:melodic-ros-core`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/cmake:apt-r32.7.1`](https://hub.docker.com/r/dustynv/cmake/tags) `(2023-09-07, 0.3GB)`<br>[`dustynv/cmake:apt-r35.2.1`](https://hub.docker.com/r/dustynv/cmake/tags) `(2023-08-29, 4.9GB)`<br>[`dustynv/cmake:apt-r35.3.1`](https://hub.docker.com/r/dustynv/cmake/tags) `(2023-09-07, 5.0GB)`<br>[`dustynv/cmake:apt-r35.4.1`](https://hub.docker.com/r/dustynv/cmake/tags) `(2023-10-07, 4.9GB)` |
| &nbsp;&nbsp;&nbsp;Notes | upgrade cmake with apt |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag cmake_apt)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host cmake_apt:35.2.1

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag cmake_apt)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag cmake_apt) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build cmake_apt
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
