# ninja

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`ninja`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dependants | [`causalconv1d:1.6.0`](/packages/ml/mamba/causalconv1d) [`cobra:0.0.1`](/packages/ml/mamba/cobra) [`cosmos-reason1`](/packages/diffusion/cosmos/cosmos-reason1) [`dimba:1.0`](/packages/ml/mamba/dimba) [`dynamo:0.3.2`](/packages/llm/dynamo/dynamo) [`fruitnerf:1.0`](/packages/3d/nerf/fruitnerf) [`genai-bench:0.1.0`](/packages/llm/sglang/genai-bench) [`genesis-world:0.2.2`](/packages/sim/genesis) [`glomap:2.0.0`](/packages/3d/3dvision/glomap) [`gsplat:1.5.3`](/packages/3d/gaussian_splatting/gsplat) [`hloc:1.4`](/packages/3d/3dvision/hloc) [`hloc:1.5`](/packages/3d/3dvision/hloc) [`hymba`](/packages/llm/hymba) [`kai_scheduler:0.5.5`](/packages/llm/dynamo/kai-scheduler) [`l4t-dynamo`](/packages/ml/l4t/l4t-dynamo) [`libcom:0.1.0`](/packages/multimedia/libcom) [`llama-factory`](/packages/llm/llama-factory) [`log-linear-attention:0.0.1`](/packages/attention/log-linear-attention) [`mamba:2.2.5`](/packages/ml/mamba/mamba) [`mambavision:1.0`](/packages/ml/mamba/mambavision) [`meshlab:MeshLab-2023.12`](/packages/3d/3dvision/meshlab) [`meshlab:MeshLab-2025.03`](/packages/3d/3dvision/meshlab) [`minference:0.1.7`](/packages/llm/minference) [`mooncake:0.3.5`](/packages/llm/dynamo/mooncake) [`nerfstudio:1.1.7`](/packages/3d/nerf/nerfstudio) [`nerfview:0.1.4`](/packages/3d/gaussian_splatting/nerfview) [`nixl:0.3.2`](/packages/llm/dynamo/nixl) [`nvidia_modelopt:0.32.0`](/packages/llm/tensorrt_optimizer/nvidia-modelopt) [`partpacker:0.1.0`](/packages/3d/3dobjects/partpacker) [`piper1-tts:1.3.0`](/packages/speech/piper1-tts) [`pixsfm:1.0`](/packages/3d/3dvision/pixsfm) [`protomotions:2.5.0`](/packages/robots/protomotions) [`pyceres:2.5`](/packages/3d/3dvision/pyceres) [`pycolmap:3.12`](/packages/3d/3dvision/pycolmap) [`pycolmap:3.13`](/packages/3d/3dvision/pycolmap) [`pymeshlab:2023.12.post2`](/packages/3d/3dvision/pymeshlab) [`pymeshlab:2023.12.post3`](/packages/3d/3dvision/pymeshlab) [`pymeshlab:2025.6.23.dev0`](/packages/3d/3dvision/pymeshlab) [`robogen`](/packages/sim/robogen) [`sapiens`](/packages/vit/sapiens) [`sgl-kernel:0.2.3`](/packages/llm/sglang/sgl-kernel) [`sglang:0.4.4`](/packages/llm/sglang) [`sglang:0.4.6`](/packages/llm/sglang) [`sglang:0.4.9`](/packages/llm/sglang) [`sparc3d:0.1.0`](/packages/3d/3dobjects/sparc3d) [`taichi:1.8.0`](/packages/sim/genesis/taichi) [`tensorrt_llm:0.12`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.22.0`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`videomambasuite:1.0`](/packages/ml/mamba/videomambasuite) [`vllm:0.7.4`](/packages/llm/vllm) [`vllm:0.8.4`](/packages/llm/vllm) [`vllm:0.9.0`](/packages/llm/vllm) [`vllm:0.9.2`](/packages/llm/vllm) [`vllm:0.9.3`](/packages/llm/vllm) [`vllm:v0.8.5.post1`](/packages/llm/vllm) [`vtk:9.3.1`](/packages/sim/genesis/vtk) [`vtk:9.4.2`](/packages/sim/genesis/vtk) [`vtk:9.5.0`](/packages/sim/genesis/vtk) [`xgrammar:0.1.15`](/packages/llm/xgrammar) [`xgrammar:0.1.18`](/packages/llm/xgrammar) [`xgrammar:0.1.19`](/packages/llm/xgrammar) [`xgrammar:0.1.20`](/packages/llm/xgrammar) [`xgrammar:0.1.21`](/packages/llm/xgrammar) [`zigma:1.0`](/packages/ml/mamba/zigma) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag ninja)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host ninja:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag ninja)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag ninja) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build ninja
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
