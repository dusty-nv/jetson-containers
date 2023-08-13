
* Talk live with LLM's using [RIVA](/packages/riva-client) ASR and TTS!
* Requires the [RIVA server](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) and [`text-generation-webui`](/packages/llm/text-generation-webui) to be running
* 

### Audio Check

First, it's recommended to test your microphone/speaker with RIVA ASR/TTS.  Follow the steps from the [`riva-client:python`](/packages/riva-client) package:

1. Start the RIVA server running on your Jetson by following [`riva_quickstart_arm64`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64)
2. List your [audio devices](/packages/riva-client/README.md#list-audio-devices)
3. Perform the ASR/TTS [loopback test](/packages/riva-client/README.md#loopback)

### Load LLM

Next, start [`text-generation-webui`](/packages/llm/text-generation-webui) with the `--api` flag and load your chat model of choice through the web UI:

```bash
./run.sh --workdir /opt/text-generation-webui $(./autotag text-generation-webui) \
   python3 server.py --listen --verbose --api \
	--model-dir=/data/models/text-generation-webui
```

Alternatively, you can manually specify the model that you want to load without needing to use the web UI:

```bash
./run.sh --workdir /opt/text-generation-webui $(./autotag text-generation-webui) \
   python3 server.py --listen --verbose --api \
	--model-dir=/data/models/text-generation-webui \
	--model=llama-2-13b-chat.ggmlv3.q4_0.bin \
	--loader=llamacpp \
	--n-gpu-layers=128
```

See here for command-line arguments:  https://github.com/oobabooga/text-generation-webui/tree/main#basic-settings