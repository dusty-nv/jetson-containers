
<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_voice_clip.gif">

> [!NOTE]  
> For llamaspeak version 2 with multimodal support, see the [`local_llm`](https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/local_llm) container

* Talk live with LLM's using [NVIDIA Riva](/packages/audio/riva-client) ASR and TTS!
* Requires the [`riva-server`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) and [`text-generation-webui`](/packages/llm/text-generation-webui) to be running

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_block_diagram.jpg">

### Start Riva

First, follow the steps from the [`riva-client:python`](/packages/audio/riva-client) package to run and test the Riva server:

1. Start the Riva server on your Jetson by following [`riva_quickstart_arm64`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64)
2. Run some of the Riva ASR examples to confirm that ASR is working:  https://github.com/nvidia-riva/python-clients#asr
3. Run some of the Riva TTS examples to confirm that TTS is working:  https://github.com/nvidia-riva/python-clients#tts

You can also see this helpful video and guide from JetsonHacks for setting up Riva:  [**Speech AI on Jetson Tutorial**](https://jetsonhacks.com/2023/08/07/speech-ai-on-nvidia-jetson-tutorial/)

### Load LLM

Next, start [`text-generation-webui`](/packages/llm/text-generation-webui) (version 1.7) with the `--api` flag and load your chat model of choice through it's web UI on port 7860:

```bash
./run.sh --workdir /opt/text-generation-webui $(./autotag text-generation-webui:1.7) \
   python3 server.py --listen --verbose --api \
	--model-dir=/data/models/text-generation-webui
```
> **note:** launch the `text-generation-webui:1.7` container to maintain API compatability

Alternatively, you can manually specify the model that you want to load without needing to use the web UI:

```bash
./run.sh --workdir /opt/text-generation-webui $(./autotag text-generation-webui:1.7) \
   python3 server.py --listen --verbose --api \
	--model-dir=/data/models/text-generation-webui \
	--model=llama-2-13b-chat.Q4_K_M.gguf \
	--loader=llamacpp \
	--n-gpu-layers=128 \
	--n_ctx=4096 \
	--n_batch=4096 \
	--threads=$(($(nproc) - 2))
```

See here for command-line arguments:  https://github.com/oobabooga/text-generation-webui/tree/main#basic-settings

### Enabling HTTPS/SSL

Browsers require HTTPS to be used in order to access the client's microphone.  Hence, you'll need to create a self-signed SSL certificate and key:

```bash
$ cd /path/to/your/jetson-containers/data
$ openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj '/CN=localhost'
```

You'll want to place these in your [`jetson-containers/data`](/data) directory, because this gets automatically mounted into the containers under `/data`, and will keep your SSL certificate persistent across container runs.  When you first navigate your browser to a page that uses these self-signed certificates, it will issue you a warning since they don't originate from a trusted authority:

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/ssl_warning.jpg" width="400">

You can choose to override this, and it won't re-appear again until you change certificates or your device's hostname/IP changes.

### Run Llamaspeak

To run the llamaspeak chat server with its default arguments and the SSL keys you generated, start it like this:

```bash
./run.sh --env SSL_CERT=/data/cert.pem --env SSL_KEY=/data/key.pem $(./autotag llamaspeak)
```

See [`chat.py`](chat.py) for command-line options that can be changed.  For example, to enable `--verbose` or `--debug` logging:

```bash
./run.sh --workdir=/opt/llamaspeak \
  --env SSL_CERT=/data/cert.pem \
  --env SSL_KEY=/data/key.pem \
  $(./autotag llamaspeak) \
  python3 chat.py --verbose
```
> if you're having issues with getting audio or responses from the web client, enable debug logging to check the message traffic.

The default port is `8050`, but that can be changed with the `--port` argument.  You can then navigate your browser to `https://HOSTNAME:8050`
