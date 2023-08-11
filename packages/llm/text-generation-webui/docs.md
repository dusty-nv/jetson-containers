
![](https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/images/text-generation-webui_sf-trip.gif)

* text-generation-webui from https://github.com/oobabooga/text-generation-webui (found under `/opt/text-generation-webui`)
* see the tutorial on the [**Jetson Generative AI Playground**](https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/tutorial_text-generation.html)

This container has a default run command that will automatically start the webserver like this:

```bash
cd /opt/text-generation-webui && python3 server.py \
  --model-dir=/data/models/text-generation-webui \
  --chat \
  --listen
```
