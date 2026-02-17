# arrow

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`arrow:20.0.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `arrow` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | installed under `/usr/local` |

| **`arrow:19.0.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dependants | [`cudf:25.08.00`](/packages/ml/rapids/cudf) [`cuml`](/packages/ml/rapids/cuml) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | installed under `/usr/local` |

| **`arrow:14.0.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | installed under `/usr/local` |

| **`arrow:12.0.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dependants | [`cudf:23.10.03`](/packages/ml/rapids/cudf) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | installed under `/usr/local` |

| **`arrow:5.0.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | installed under `/usr/local` |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag arrow)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host arrow:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag arrow)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag arrow) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build arrow
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
