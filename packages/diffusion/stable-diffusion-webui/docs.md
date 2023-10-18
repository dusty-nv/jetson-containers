
<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/diffusion_webui.jpg">

* stable-diffusion-webui from https://github.com/AUTOMATIC1111/stable-diffusion-webui (found under `/opt/stable-diffusion-webui`)
* with TensorRT extension from https://github.com/AUTOMATIC1111/stable-diffusion-webui-tensorrt 
* see the tutorial at the [**Jetson Generative AI Lab**](https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/tutorial_diffusion.html)

This container has a default run command that will automatically start the webserver like this:

```bash
cd /opt/stable-diffusion-webui && python3 launch.py \
  --data=/data/models/stable-diffusion \
  --enable-insecure-extension-access \
  --xformers \
  --listen \
  --port=7860
```

After starting the container, you can navigate your browser to `http://$IP_ADDRESS:7860` (substitute the address or hostname of your device).  The server will automatically download the default model ([`stable-diffusion-1.5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)) during startup.

Other configuration arguments can be found at [AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings)

* `--medvram` (sacrifice some performance for low VRAM usage)
* `--lowvram` (sacrafice a lot of speed for very low VRAM usage)

See the [`stable-diffusion`](/packages/diffusion/stable-diffusion) container to run image generation from a script (`txt2img.py`) as opposed to the web UI. 

### Tips & Tricks

Negative prompts:  https://huggingface.co/spaces/stabilityai/stable-diffusion/discussions/7857

Stable Diffusion XL
  * https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
  * https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
  * https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
  * https://stable-diffusion-art.com/sdxl-model/