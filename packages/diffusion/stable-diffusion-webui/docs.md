
![two robots sitting by a lake by a mountain](/docs/images/diffusion_webui.jpg)

* stable-diffusion-webui: https://github.com/AUTOMATIC1111/stable-diffusion-webui (`/opt/stable-diffusion-webui`)
* with TensorRT extension: https://github.com/AUTOMATIC1111/stable-diffusion-webui-tensorrt 
* faster performance than the base [`stable-diffusion`](/packages/diffusion/stable-diffusion) package (`txt2img.py`)
* tested on default `stable-diffusion-1.5` model: https://huggingface.co/runwayml/stable-diffusion-v1-5

This container has a default run command that will automatically start the webserver like this:

```bash
cd /opt/stable-diffusion-webui && python3 launch.py \
	--data=/data/models/stable-diffusion \
	--enable-insecure-extension-access \
	--xformers \
	--listen \
	--port=7860
```

After starting the container, you can navigate your browser to `http://$IP_ADDRESS:7860` (substitute the address or hostname of your device for `$IP_ADDRESS`).  It will automatically download the diffusion model when starting.

Other configuration arguments can be found at https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings

* `--medvram` (sacrifice some performance for low VRAM usage)
* `--lowvram` (sacrafice a lot of speed for very low VRAM usage)
