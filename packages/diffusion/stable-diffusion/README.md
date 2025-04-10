# stable-diffusion

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/diffusion_astronaut.jpg">

Generate images from text (txt2img) or from other images (img2img)

* stable-diffusion: https://github.com/CompVis/stable-diffusion (`/opt/stable-diffusion`)
* with memory optimizations: https://github.com/basujindal/stable-diffusion (`/opt/stable-diffusion/optimizedSD`)
* tested on `stable-diffusion-1.4` model: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original

See the [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) container for a faster implementation with a web interface.

### txt2img

Download the [stable-diffusion-1.4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) model (`sd-v1-4.ckpt`)

```bash
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O /data/models/stable-diffusion/sd-v1-4.ckpt
```

Then run this in the container to generate images (by default, six 512x512 images with 50 refinement steps)

```bash
cd /opt/stable-diffusion && python3 scripts/txt2img.py --plms \
  --ckpt /data/models/stable-diffusion/sd-v1-4.ckpt \
  --outdir /data/images/stable-diffusion \
  --prompt "a photograph of an astronaut riding a horse"
```

See here for options:  https://github.com/CompVis/stable-diffusion#reference-sampling-script

For just one 512x512 image with 25 steps:

```bash
cd /opt/stable-diffusion && python3 scripts/txt2img.py --plms \
  --n_samples 1 --n_iter 1 --ddim_steps 25 \
  --ckpt /data/models/stable-diffusion/sd-v1-4.ckpt \
  --outdir /data/images/stable-diffusion \
  --prompt "two robots walking in the woods"
```

* Change the image resolution with `--W` and `--H` (the default is 512x512)
* Change the `--seed` to have the images be different (the default seed is 42)

For Jetson Orin Nano and reduced memory usage:

```bash
cd /opt/stable-diffusion && python3 optimizedSD/optimized_txt2img.py \
  --sampler plms --seed 42 \
  --n_samples 1 --n_iter 1 --ddim_steps 25 \
  --ckpt /data/models/stable-diffusion/sd-v1-4.ckpt \
  --outdir /data/images/stable-diffusion \
  --prompt "a photograph of an astronaut riding a horse"
```

To run all these steps from a script, see [`stable-diffusion/test.sh`](/packages/diffusion/stable-diffusion/test.sh) 
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`stable-diffusion`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`stable-diffusion_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/stable-diffusion_jp51.yml?label=stable-diffusion:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/stable-diffusion_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch:2.2`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-diffusion`](/packages/l4t/l4t-diffusion) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/stable-diffusion:r35.2.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) `(2023-12-14, 6.1GB)`<br>[`dustynv/stable-diffusion:r35.3.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) `(2023-12-12, 6.1GB)`<br>[`dustynv/stable-diffusion:r35.4.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) `(2023-12-15, 6.1GB)` |
| &nbsp;&nbsp;&nbsp;Notes | disabled on JetPack 4 |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/stable-diffusion:r35.2.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) | `2023-12-14` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/stable-diffusion:r35.3.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) | `2023-12-12` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/stable-diffusion:r35.4.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) | `2023-12-15` | `arm64` | `6.1GB` |

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
jetson-containers run $(autotag stable-diffusion)

# or explicitly specify one of the container images above
jetson-containers run dustynv/stable-diffusion:r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/stable-diffusion:r35.4.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag stable-diffusion)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag stable-diffusion) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build stable-diffusion
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
