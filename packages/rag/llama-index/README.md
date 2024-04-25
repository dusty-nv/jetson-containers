# llama-index

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* llama-index from https://www.llamaindex.ai/

### Starting llama-index container (only)

```bash
jetson-containers run $(./autotag llama-index)
```

### Running a starter RAG example with Ollama

This is based on the [official tutorial for local models](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/).

#### Data set up for the sample

On the Docker host console, copy the L4T-README text files to jetson-container's `/data` directory.

```bash
cd jetson-containers
mkdir -p data/documents/paul_grapham
wget "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1" -O data/documents/paul_grapham/paul_graham_essay.txt
mkdir -p data/documents/L4T-README
cp /media/jetson/L4T-README/*.txt data/documents/L4T-README/
```

#### Docker-compose to run llama_index container with ollama container

> Here assumes we are on JetPack 6.0 DP and have followed the instruction [here](https://www.jetson-ai-lab.com/tips_ssd-docker.html#docker) for installing Docker.

Move to the `llama-index` package directory where `compose.yml` is saved, and use docker compose to run two containers.

```bash
cd ./packages/llm/llama-index
docker compose up
```

Open a new terminal and attach to the llama_index container.

```bash
docker exec -it llama-index bash
```

Once in the llama_index container, first download the Llama2 model using `ollama` command.

```bash
ollama pull llama2
```

Then, run the sample script to ask Jetson related questions (***"With USB device mode, what IP address Jetson gets? Which file should be edited in order to change the IP address assigned to Jetson?"***)to let the Llama-2 model answer based on the provided README files.

```bash
python3 samples/llamaindex_starter.py
```
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`llama-index`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`ollama`](/packages/llm/ollama) [`rust`](/packages/build/rust) [`jupyterlab`](/packages/jupyterlab) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag llama-index)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host llama-index:36.2.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag llama-index)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag llama-index) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build llama-index
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
