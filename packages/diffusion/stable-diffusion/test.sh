#!/usr/bin/env bash
echo "testing stable-diffusion (txt2img)"

MODEL="/data/models/stable-diffusion/sd-v1.4.ckpt"
PROMPT="a photograph of an astronaut riding a horse"

# download model
mkdir -p /data/models/stable-diffusion /data/images/stable-diffusion || "data dir(s) already exist"

if [ ! -f "$MODEL" ]; then
	wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O "$MODEL"
fi

# txt2img needs to be run from it's directory
cd /opt/stable-diffusion

# generate one image, 25 steps (default is 6 512x512 images, 50 steps)
/usr/bin/time -v python3 scripts/txt2img.py --plms \
  --n_samples 1 --n_iter 1 --ddim_steps 25 \
  --outdir /data/images/stable-diffusion \
  --ckpt "$MODEL" \
  --prompt "$PROMPT"

# now try it with the memory optimizations
/usr/bin/time -v python3 optimizedSD/optimized_txt2img.py \
  --sampler plms --seed 42 \
  --n_samples 1 --n_iter 1 --ddim_steps 25 \
  --outdir /data/images/stable-diffusion \
  --ckpt "$MODEL" \
  --prompt "$PROMPT"