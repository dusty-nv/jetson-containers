# homeassistant-base

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`homeassistant-base:master`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `homeassistant-base` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) |
| &nbsp;&nbsp;&nbsp;Dependants | [`homeassistant-core:2025.7.1`](/packages/smart-home/homeassistant-core) [`wyoming-assist-microphone:1.4.1`](/packages/smart-home/wyoming/wyoming-assist-microphone) [`wyoming-assist-microphone:master`](/packages/smart-home/wyoming/wyoming-assist-microphone) [`wyoming-openwakeword:1.10.0`](/packages/smart-home/wyoming/wyoming-openwakeword) [`wyoming-openwakeword:master`](/packages/smart-home/wyoming/wyoming-openwakeword) [`wyoming-piper:1.6.2`](/packages/smart-home/wyoming/wyoming-piper1) [`wyoming-whisper:2.5.0`](/packages/smart-home/wyoming/wyoming-whisper) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The `homeassistant` base ubuntu image with pre-installed dependencies based on `https://github.com/home-assistant/docker-base/blob/master/ubuntu/Dockerfile` |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag homeassistant-base)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host homeassistant-base:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag homeassistant-base)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag homeassistant-base) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build homeassistant-base
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
