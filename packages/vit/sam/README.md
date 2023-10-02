# sam

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* Segment Anything from https://github.com/facebookresearch/segment-anything

The `sam` container has a default run command to launch Jupyter Lab with notebook directory to be `/opt/`

Use your web browser to access `http://HOSTNAME:8888`

### How to run Jupyter notebooks

Once you are on Jupyter Lab site, navigate to `notebooks` directory.

#### Automatic Mask Generator Example notebook

Open `automatic_mask_generator_example.ipynb`.

Create a cell below the 4th cell, with only the following line and execute.

```
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Then, start executing the following cells (cells below **Set-up**)

#### Predictor Example notebook

Open `predictor_example.ipynb`.

Make sure you have `sam_vit_h_4b8939.pth` checkpoint file saved under `notebooks` directory.

Then, start executing the cells below **Set-up**.

### Benchmark script

You can run the following command to run a benchmark script.

```
python3 benchmark.py --save sam.csv
```

Or for full options:

```
python3 benchmark.py \
  --images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg  https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg \
  --runs=1 --warmup=0 \
  --save sam.csv
```

Outputs are:

- `sam_benchmark_output.jpg` :
- `sam.csv` (optional) : 

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`sam`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`sam_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/sam_jp51.yml?label=sam:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/sam_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`onnxruntime`](/packages/onnxruntime) [`rust`](/packages/rust) [`jupyterlab`](/packages/jupyterlab) |
| &nbsp;&nbsp;&nbsp;Dependants | [`tam`](/packages/vit/tam) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/sam:r35.2.1`](https://hub.docker.com/r/dustynv/sam/tags) `(2023-10-01, 6.1GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/sam:r35.2.1`](https://hub.docker.com/r/dustynv/sam/tags) | `2023-10-01` | `arm64` | `6.1GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag sam)

# or explicitly specify one of the container images above
./run.sh dustynv/sam:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/sam:r35.2.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag sam)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag sam) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh sam
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
