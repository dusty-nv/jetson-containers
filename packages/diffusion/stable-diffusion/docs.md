
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