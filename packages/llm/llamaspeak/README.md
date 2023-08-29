# llamaspeak

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* Talk live with LLM's using [RIVA](/packages/riva-client) ASR and TTS!
* Requires the [RIVA server](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) and [`text-generation-webui`](/packages/llm/text-generation-webui) to be running
* 

### Audio Check

First, it's recommended to test your microphone/speaker with RIVA ASR/TTS.  Follow the steps from the [`riva-client:python`](/packages/riva-client) package:

1. Start the RIVA server running on your Jetson by following [`riva_quickstart_arm64`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64)
2. List your [audio devices](/packages/riva-client/README.md#list-audio-devices)
3. Perform the ASR/TTS [loopback test](/packages/riva-client/README.md#loopback)

### Load LLM

Next, start [`text-generation-webui`](/packages/llm/text-generation-webui) with the `--api` flag and load your chat model of choice through the web UI:

```bash
./run.sh --workdir /opt/text-generation-webui $(./autotag text-generation-webui) \
   python3 server.py --listen --verbose --api \
	--model-dir=/data/models/text-generation-webui
```

Alternatively, you can manually specify the model that you want to load without needing to use the web UI:

```bash
./run.sh --workdir /opt/text-generation-webui $(./autotag text-generation-webui) \
   python3 server.py --listen --verbose --api \
	--model-dir=/data/models/text-generation-webui \
	--model=llama-2-13b-chat.ggmlv3.q4_0.bin \
	--loader=llamacpp \
	--n-gpu-layers=128
```

See here for command-line arguments:  https://github.com/oobabooga/text-generation-webui/tree/main#basic-settings
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`llamaspeak`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`riva-client:python`](/packages/riva-client) [`numpy`](/packages/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag llamaspeak)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host llamaspeak:35.2.1

```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag llamaspeak)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag llamaspeak) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh llamaspeak
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
