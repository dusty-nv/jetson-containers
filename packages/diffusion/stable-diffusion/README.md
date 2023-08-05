# stable-diffusion

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


![a photograph of an astronaut riding a horse](/docs/images/diffusion_astronaut.jpg)

Generate images from text (txt2img) or from other images (img2img)

* stable-diffusion: https://github.com/CompVis/stable-diffusion (`/opt/stable-diffusion`)
* with memory optimizations: https://github.com/basujindal/stable-diffusion (`/opt/stable-diffusion/optimizedSD`)
* tested on `stable-diffusion-1.4` model: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original

See the [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) container for a faster implementation with a web interface.

### txt2img

Download the [stable-diffusion-1.4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) model (`sd-v1-4.ckpt`)

```bash
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O /data/models/stable-diffusion
```

Then run this in the container to generate images (by default, six 512x512 images with 50 refinement steps)

```bash
cd /opt/stable-diffusion
python3 scripts/txt2img.py --plms \
  --ckpt /data/models/stable-diffusion/sd-v1-4.ckpt \
  --outdir /data/images/stable-diffusion \
  --prompt "a photograph of an astronaut riding a horse"
```

See here for options:  https://github.com/CompVis/stable-diffusion#reference-sampling-script

For just one 512x512 image with 25 steps:

```bash
python3 scripts/txt2img.py --plms \
  --n_samples 1 --n_iter 1 --ddim_steps 25 \
  --ckpt /data/models/stable-diffusion/sd-v1-4.ckpt \
  --outdir /data/images/stable-diffusion \
  --prompt "two robots walking in the woods"
```

* Change the image resolution with `--W` and `--H` (the default is 512x512)
* Change the `--seed` to have the images be different (the default seed is 42)

For Jetson Orin Nano and reduced memory usage:

```bash
python3 optimizedSD/optimized_txt2img.py \
  --sampler plms --seed 42 \
  --n_samples 1 --n_iter 1 --ddim_steps 25 \
  --ckpt /data/models/stable-diffusion/sd-v1-4.ckpt \
  --outdir /data/images/stable-diffusion \
  --prompt "a photograph of an astronaut riding a horse"
```

To run these steps from a script, see [`stable-diffusion/test.sh`](/packages/diffusion/stable-diffusion/test.sh) 
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`stable-diffusion`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`stable-diffusion_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/stable-diffusion_jp51.yml?label=stable-diffusion:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/stable-diffusion_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`bitsandbytes`](/packages/llm/bitsandbytes) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-diffusion`](/packages/l4t/l4t-diffusion) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/stable-diffusion:r35.2.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) `(2023-08-04, 5.7GB)`<br>[`dustynv/stable-diffusion:r35.3.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) `(2023-08-04, 5.7GB)`<br>[`dustynv/stable-diffusion:r35.4.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) `(2023-08-04, 5.7GB)` |
| &nbsp;&nbsp;&nbsp;Notes | disabled on JetPack 4 |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/stable-diffusion:r35.2.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) | `2023-08-04` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/stable-diffusion:r35.3.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) | `2023-08-04` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/stable-diffusion:r35.4.1`](https://hub.docker.com/r/dustynv/stable-diffusion/tags) | `2023-08-04` | `arm64` | `5.7GB` |

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
./run.sh $(./autotag stable-diffusion)

# or explicitly specify one of the container images above
./run.sh dustynv/stable-diffusion:r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/stable-diffusion:r35.4.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag stable-diffusion)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag stable-diffusion) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh stable-diffusion
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
