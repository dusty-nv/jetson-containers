# lerobot

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

* lerobot project from https://github.com/huggingface/lerobot/
* see the tutorial at [**Jetson AI Lab**](https://jetson-ai-lab.com/tutorial_lerobot.html) (Coming scoon)

## Usage Examples

### Example: Visualize datasets

*Check the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#visualize-datasets).*

On Docker host (Jetson native), first launch rerun.io.

```bash
pip install rerun-sdk
rerun
```

Then, start the docker container to run the visualization script.

```bash
jetson-containers run $(autotag lerobot) python3 lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

### Example: Evaluate a pretrained policy

*Check the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#evaluate-a-pretrained-policy).*

```bash
jetson-containers run $(autotag lerobot) python3 lerobot/scripts/eval.py \
    -p lerobot/diffusion_pusht \
    eval.n_episodes=10 \
    eval.batch_size=10
```

### Example: Train your own policy

*Check the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#train-your-own-policy).*

```bash
jetson-containers run --shm-size=6g $(autotag lerobot) python3 lerobot/scripts/train.py \
    policy=act \
    env=aloha \
    env.task=AlohaInsertion-v0 \
    dataset_repo_id=lerobot/aloha_sim_insertion_human 
```

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`lerobot`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `lerobot` |
| &nbsp;&nbsp;&nbsp;Builds | [![`lerobot:r36.3.0-cu122`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/lerobot_jp60.yml?label=lerobot:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/lerobot_jp60.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`torchvision`](/packages/pytorch/torchvision/) [`opencv`](/packages/opencv) [`huggingface_hub`](/packages/llm/huggingface_hub) [`h5py`](/packages/build/h5py) |
| &nbsp;&nbsp;&nbsp;Dependants |  |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images |  |
| &nbsp;&nbsp;&nbsp;Notes |  |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/lerobot:r36.3.0-cu122`](https://hub.docker.com/r/dustynv/lerobot/tags) | `2024-xx-xx` | `arm64` | `nnn GB` |


</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag lerobot)

# or explicitly specify one of the container images above
jetson-containers run dustynv/lerobot:r36.3.0-cu122

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/lerobot:r36.3.0-cu122
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag lerobot)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag lerobot) python3 lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build lerobot
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
