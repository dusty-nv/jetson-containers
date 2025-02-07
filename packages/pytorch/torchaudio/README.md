# torchaudio

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`torchaudio:2.5.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `torchaudio` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==36.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.5`](/packages/pytorch) |
| &nbsp;&nbsp;&nbsp;Dependants | [`ai-toolkit`](/packages/diffusion/ai-toolkit) [`audiocraft`](/packages/speech/audiocraft) [`causalconv1d:1.4.0`](/packages/ml/mamba/causalconv1d) [`causalconv1d:1.4.0-builder`](/packages/ml/mamba/causalconv1d) [`causalconv1d:1.6.0`](/packages/ml/mamba/causalconv1d) [`causalconv1d:1.6.0-builder`](/packages/ml/mamba/causalconv1d) [`cobra:0.0.1`](/packages/ml/mamba/cobra) [`comfyui`](/packages/diffusion/comfyui) [`cosmos`](/packages/diffusion/cosmos) [`dimba:1.0`](/packages/ml/mamba/dimba) [`fruitnerf:1.0`](/packages/nerf/fruitnerf) [`gsplat:1.3.0`](/packages/nerf/gsplat) [`gsplat:1.3.0-builder`](/packages/nerf/gsplat) [`gsplat:1.5.0`](/packages/nerf/gsplat) [`gsplat:1.5.0-builder`](/packages/nerf/gsplat) [`hymba`](/packages/llm/hymba) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`llama-factory`](/packages/llm/llama-factory) [`local_llm`](/packages/llm/local_llm) [`mamba:2.2.2`](/packages/ml/mamba/mamba) [`mamba:2.2.2-builder`](/packages/ml/mamba/mamba) [`mamba:2.2.5`](/packages/ml/mamba/mamba) [`mamba:2.2.5-builder`](/packages/ml/mamba/mamba) [`mambavision:1.0`](/packages/ml/mamba/mambavision) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`nemo:1.23.0`](/packages/llm/nemo) [`nemo:2.0.0`](/packages/llm/nemo) [`nemo:2.0.0-builder`](/packages/llm/nemo) [`nerfacc:0.5.3`](/packages/nerf/nerfacc) [`nerfacc:0.5.3-builder`](/packages/nerf/nerfacc) [`nerfacc:0.5.4`](/packages/nerf/nerfacc) [`nerfacc:0.5.4-builder`](/packages/nerf/nerfacc) [`nerfstudio:1.1.4`](/packages/nerf/nerfstudio) [`nerfstudio:1.1.4-builder`](/packages/nerf/nerfstudio) [`nerfstudio:1.1.5`](/packages/nerf/nerfstudio) [`nerfstudio:1.1.5-builder`](/packages/nerf/nerfstudio) [`nerfstudio:1.1.6`](/packages/nerf/nerfstudio) [`nerfstudio:1.1.6-builder`](/packages/nerf/nerfstudio) [`onnxruntime_genai:0.6.0`](/packages/ml/onnxruntime_genai) [`onnxruntime_genai:0.6.0-builder`](/packages/ml/onnxruntime_genai) [`prismatic`](/packages/vlm/prismatic) [`sapiens`](/packages/vit/sapiens) [`sglang`](/packages/llm/sglang) [`videomambasuite:1.0`](/packages/ml/mamba/videomambasuite) [`vllm:0.7.2`](/packages/llm/vllm) [`vllm:0.7.2-builder`](/packages/llm/vllm) [`voicecraft`](/packages/speech/voicecraft) [`whisper`](/packages/speech/whisper) [`whisperx`](/packages/speech/whisperx) [`xtts`](/packages/speech/xtts) [`zigma:1.0`](/packages/ml/mamba/zigma) [`zigma:1.0-builder`](/packages/ml/mamba/zigma) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/torchaudio:r32.7.1`](https://hub.docker.com/r/dustynv/torchaudio/tags) | `2023-12-14` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/torchaudio:r35.2.1`](https://hub.docker.com/r/dustynv/torchaudio/tags) | `2023-12-14` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/torchaudio:r35.3.1`](https://hub.docker.com/r/dustynv/torchaudio/tags) | `2023-12-11` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/torchaudio:r35.4.1`](https://hub.docker.com/r/dustynv/torchaudio/tags) | `2023-12-12` | `arm64` | `5.4GB` |

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
jetson-containers run $(autotag torchaudio)

# or explicitly specify one of the container images above
jetson-containers run dustynv/torchaudio:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/torchaudio:r35.2.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag torchaudio)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag torchaudio) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build torchaudio
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
