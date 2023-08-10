
* text-generation-webui: https://github.com/oobabooga/text-generation-webui (`/opt/text-generation-webui`)
* supporting ExLlama loader: https://github.com/turboderp/exllama

This container has a default run command that will automatically start the webserver like this:

```bash
cd /opt/text-generation-webui && python3 server.py \
  --model-dir=/data/models/text-generation-webui \
  --chat \
  --listen
```
