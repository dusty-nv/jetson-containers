Thanks to [Manuel Schweiger](https://github.com/mschweig) for contributing this container for [RoboPoint](https://robo-point.github.io/)!

### Start Server

This container will automatically run the CMD [`/opt/robopoint/start-server.sh`](start-server.sh) upon startup, which unless overridden with a different CMD, will first download the model specified by `ROBOPOINT_MODEL` environment variable (by default [`wentao-yuan/robopoint-v1-vicuna-v1.5-13b`](https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-13b)), and then load the model in the precision set by `ROBOPOINT_QUANTIZATION` (by default `int4`)

```bash
# running this container will download the model and start the server
jetson-containers run dustynv/robopoint:r36.4.0

# set ROBOPOINT_MODEL to specify HF model to download from @wentao-yuan (or local path)
# set ROBOPOINT_QUANTIZATION to int4/int8/fp16 (default is int4, with bitsandbytes --load_in_4bit)
jetson-containers run \
  -e ROBOPOINT_MODEL="wentao-yuan/robopoint-v1-vicuna-v1.5-13b" \
  -e ROBOPOINT_QUANTIZATION="int4" \
  dustynv/robopoint:r36.4.0
```

To override the default CMD and manually set flags to the model loader:

```bash
# for these flags, run 'python3 -m robopoint.serve.model_worker --help'
jetson-containers run \
  dustynv/robopoint:r36.4.0 \
    /opt/robopoint/start-server.sh --max-len 512
```

Extra flags to the startup script get appended to the `robopoint.serve.model_worker` command-line.

### Gradio UI

Launching the server above will also start a gradio web UI, reachable at `http://JETSON_IP:7860`

`<TODO SCREENSHOT>`

### Test Client

Although you can `import robopoint` into a Python script inside the container environment that loads & performs inference with the model directly, by default RoboPoint uses a client/server architecture similar in effect to LLM [`chat.completion`] microservices due to the model sizes and dependencies.  

The [`client.py`](client.py) uses REST requests to example processes a test image, and can be run inside or outside of container.  Since the heavy lifting is done inside the server, the client has lightweight dependencies (just install `pip install gradio_client` first if running this outside of container)

```bash
# mount in the examples so they can be edited from outside container
jetson-containers run \
  -v $(jetson-containers root)/packages/robots/robopoint:/mnt \
  dustynv/robopoint:r36.4.0 \
    python3 /mnt/client.py
```

The performance is currently ~2 seconds/image on AGX Orin with int4 (bitsandbytes), which is currently fine for initial experimentation before migrating to more intensive VLM optimizations (for example in NanoLLM or SGLang), and also is appropriate for use of the REST API to save time during the frequent reloads of the clientside logic related to the robotics or simulator integration that typically occur under development.