> Thanks to [Manuel Schweiger](https://github.com/mschweig) for contributing this container for [RoboPoint](https://robo-point.github.io/)!

### Start Server

```bash
# running this container will download the model and start the server
jetson-containers run dustynv/robopoint:r36.4.0

# use ROBOPOINT_MODEL to specify HF model to download (or local path)
jetson-containers run \
  -e ROBOPOINT_MODEL="wentao-yuan/robopoint-v1-vicuna-v1.5-13b" \
  dustynv/robopoint:r36.4.0
```

### Test Client

The [`client.py`](client.py) example processes a test image, and can be run inside or outside of container.  Since the heavy lifting is done inside the server, the client has lightweight dependencies (install `pip install gradio_client` first if running this outside of container)

```bash
# mount in the examples so they can be edited from outside container
jetson-containers run \
  -v $(jetson-containers root)/packages/robots/robopoint:/mnt \
  dustynv/robopoint:r36.4.0 \
    python3 /mnt/client.py
```