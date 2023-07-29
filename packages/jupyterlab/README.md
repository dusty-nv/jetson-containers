# jupyterlab

<details open>
<summary><b>CONTAINERS</b></summary>
<br>

| **`jupyterlab`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`jupyterlab_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jupyterlab_jp46.yml?label=jupyterlab_jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jupyterlab_jp46.yml) [![`jupyterlab_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jupyterlab_jp51.yml?label=jupyterlab_jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jupyterlab_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`rust`](/packages/rust) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-ml`](/packages/l4t/l4t-ml) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | will autostart Jupyter server on port 8888 unless container entry CMD is overridden |

</details>

<details open>
<summary><b>RUN CONTAINER</b></summary>
<br>

To start the container, you can use the [`run.sh`](/run.sh)/[`autotag`](/autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag jupyterlab)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host jupyterlab:35.4.1

```
> <sup>[`run.sh`](/run.sh) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag jupyterlab)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag jupyterlab) my_app --abc xyz
```
You can pass any options to `run.sh` that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b>BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do this System Setup, then run:
```bash
./build.sh jupyterlab
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
