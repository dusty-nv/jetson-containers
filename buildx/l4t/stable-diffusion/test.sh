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

mem_capacity=$(grep MemTotal /proc/meminfo | awk '{print $2}')
echo "memory capacity:  $mem_capacity KB"

# generate one image, 25 steps (default is 6 512x512 images, 50 steps)
# (this way consumes ~12GB memory and doesn't run on 8GB boards)
if [ $mem_capacity -gt 8388608 ]; then
	echo "running scripts/txt2img.py"
	
	/usr/bin/time -v python3 scripts/txt2img.py --plms \
	  --n_samples 1 --n_iter 1 --ddim_steps 25 \
	  --outdir /data/images/stable-diffusion \
	  --ckpt "$MODEL" \
	  --prompt "$PROMPT"
fi

# now try it with the memory optimizations
# (this way consumes ~10GB memory and should run on 8GB boards)
echo "optimizedSD/optimized_txt2img.py"

/usr/bin/time -v python3 optimizedSD/optimized_txt2img.py \
  --sampler plms --seed 42 \
  --n_samples 1 --n_iter 1 --ddim_steps 25 \
  --outdir /data/images/stable-diffusion \
  --ckpt "$MODEL" \
  --prompt "$PROMPT"