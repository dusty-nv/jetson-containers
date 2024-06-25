
* NanoOWL from https://github.com/NVIDIA-AI-IOT/nanoowl/

### Run the basic usage example and copy the result to host

```
jetson-containers run --workdir /opt/nanoowl/examples \
  $(autotag nanoowl) \
    python3 owl_predict.py \
      --prompt="[an owl, a glove]" \
      --threshold=0.1 \
      --image_encoder_engine=../data/owl_image_encoder_patch32.engine
```

### Run the tree prediction example (live camera)

1. First, Ensure you have a USB webcam device connected to your Jetson.

2. Launch the demo

```
jetson-containers run --workdir /opt/nanoowl/examples/tree_demo \
  $(autotag nanoowl) \
    python3 tree_demo.py ../../data/owl_image_encoder_patch32.engine
```

3. Second, open your browser to `http://<ip address>:7860`

> You can use a PC (or any machine) to open a web browser as long as  can access the Jetson via the network

4. Type whatever prompt you like to see what works! Here are some examples

  - Example: `[a face [a nose, an eye, a mouth]]`
  - Example: `[a face (interested, yawning / bored)]`
  - Example: `(indoors, outdoors)`
