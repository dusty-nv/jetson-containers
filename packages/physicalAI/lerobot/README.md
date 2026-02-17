# lerobot

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


<img src="https://github.com/user-attachments/assets/6a8967e1-f9dd-463f-906b-d9fd1f44450f">

* LeRobot project from https://github.com/huggingface/lerobot/

## Usage Examples

### Example: Visualize datasets

On Docker host (Jetson native), first launch rerun.io (check the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#visualize-datasets))

```bash
pip install rerun-sdk
rerun
```

Then, start the docker container to run the visualization script.

```bash
jetson-containers run --shm-size=4g -w /opt/lerobot $(autotag lerobot) \
  python3 lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

### Example: Evaluate a pretrained policy

See the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#evaluate-a-pretrained-policy).

```bash
jetson-containers run --shm-size=4g -w /opt/lerobot $(autotag lerobot) \
  python3 lerobot/scripts/eval.py \
    -p lerobot/diffusion_pusht \
    eval.n_episodes=10 \
    eval.batch_size=10
```

### Example: Train your own policy

See the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#train-your-own-policy).

```bash
jetson-containers run --shm-size=4g -w /opt/lerobot $(autotag lerobot) \
  python3 lerobot/scripts/train.py \
    policy=act \
    env=aloha \
    env.task=AlohaInsertion-v0 \
    dataset_repo_id=lerobot/aloha_sim_insertion_human 
```

## Usage with Real-World Robot (Koch v1.1)

### Before starting the container : Set udev rule

On Jetson host side, we set an udev rule so that arms always get assigned the same device name as following.

- `/dev/ttyACM_kochleader`   : Leader arm
- `/dev/ttyACM_kochfollower` : Follower arm

First only connect the leader arm to Jetson and record the serial ID by running the following:

```bash
ll /dev/serial/by-id/
```

The output should look like this.

```bash
lrwxrwxrwx 1 root root 13 Sep 24 13:07 usb-ROBOTIS_OpenRB-150_BA98C8C350304A46462E3120FF121B06-if00 -> ../../ttyACM1
```

Then edit the first line of `./99-usb-serial.rules` like the following.

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="2f5d", ATTRS{idProduct}=="2202", ATTRS{serial}=="BA98C8C350304A46462E3120FF121B06", SYMLINK+="ttyACM_kochleader"
SUBSYSTEM=="tty", ATTRS{idVendor}=="2f5d", ATTRS{idProduct}=="2202", ATTRS{serial}=="00000000000000000000000000000000", SYMLINK+="ttyACM_kochfollower"
```

Now disconnect the leader arm, and then only connect the follower arm to Jetson.

Repeat the same steps to record the serial to edit the second line of `99-usb-serial.rules` file.

```bash
$ ll /dev/serial/by-id/
lrwxrwxrwx 1 root root 13 Sep 24 13:07 usb-ROBOTIS_OpenRB-150_483F88DC50304A46462E3120FF0C081A-if00 -> ../../ttyACM0
$ vi ./data/lerobot/99-usb-serial.rules
```

You should have `./99-usb-serial.rules` now looking like this:

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="2f5d", ATTRS{idProduct}=="2202", ATTRS{serial}=="BA98C8C350304A46462E3120FF121B06", SYMLINK+="ttyACM_kochleader"
SUBSYSTEM=="tty", ATTRS{idVendor}=="2f5d", ATTRS{idProduct}=="2202", ATTRS{serial}=="483F88DC50304A46462E3120FF0C081A", SYMLINK+="ttyACM_kochfollower"
```

Finally copy this under `/etc/udev/rules.d/` (of host), and restart Jetson.

```
sudo cp ./99-usb-serial.rules /etc/udev/rules.d/
sudo reboot
```

After reboot, check if we now have achieved the desired fixed simlinks names for the arms.

```bash
ls -l /dev/ttyACM*
```

You should get something like this:

```bash
crw-rw---- 1 root dialout 166, 0 Sep 24 17:20 /dev/ttyACM0
crw-rw---- 1 root dialout 166, 1 Sep 24 16:13 /dev/ttyACM1
lrwxrwxrwx 1 root root         7 Sep 24 17:20 /dev/ttyACM_kochfollower -> ttyACM0
lrwxrwxrwx 1 root root         7 Sep 24 16:13 /dev/ttyACM_kochleader -> ttyACM1
```

### Create the local copy of lerobot on host (under `jetson-containers/data` dir)

```bash
cd jetson-containers
./packages/robots/lerobot/clone_lerobot_dir_under_data.sh
./packages/robots/lerobot/copy_overlay_files_in_data_lerobot.sh
```

### Start the container with local lerobot dir mounted

```bash
./run.sh \
  --csi2webcam --csi-capture-res='1640x1232@30' --csi-output-res='640x480@30' \
  -v ${PWD}/data/lerobot/:/opt/lerobot/ \
  $(./autotag lerobot)
```

### Test with Koch arms

You will now use your local PC to access the Jupyter Lab server running on Jetson on the same network.

Once the contianer starts, you should see lines like this printed.

```
JupyterLab URL:   http://10.110.51.21:8888 (password "nvidia")
JupyterLab logs:  /data/logs/jupyter.log
```

Copy and paste the address on your web browser and access the Jupyter Lab server.

Navigate to `./notebooks/` and open the first notebook.

Now follow the Jupyter notebook contents.




<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`lerobot`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`opengl`](/packages/multimedia/opengl) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg:git`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pyav`](/packages/multimedia/pyav) [`h5py`](/packages/build/h5py) [`diffusers`](/packages/diffusion/diffusers) [`tensorrt`](/packages/cuda/tensorrt) [`cuda-python`](/packages/cuda/cuda-python) [`pycuda`](/packages/cuda/pycuda) [`jupyterlab:latest`](/packages/code/jupyterlab) [`jupyterlab:myst`](/packages/code/jupyterlab) |
| &nbsp;&nbsp;&nbsp;Dependants | [`openpi`](/packages/robots/openpi) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/lerobot:r36.3.0`](https://hub.docker.com/r/dustynv/lerobot/tags) `(2024-10-15, 7.6GB)`<br>[`dustynv/lerobot:r36.4.0`](https://hub.docker.com/r/dustynv/lerobot/tags) `(2024-12-13, 6.1GB)`<br>[`dustynv/lerobot:r36.4.0-20250305`](https://hub.docker.com/r/dustynv/lerobot/tags) `(2025-03-05, 9.4GB)`<br>[`dustynv/lerobot:r36.4.0-cu128`](https://hub.docker.com/r/dustynv/lerobot/tags) `(2025-02-09, 8.5GB)`<br>[`dustynv/lerobot:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/lerobot/tags) `(2025-04-24, 7.0GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/lerobot:r36.3.0`](https://hub.docker.com/r/dustynv/lerobot/tags) | `2024-10-15` | `arm64` | `7.6GB` |
| &nbsp;&nbsp;[`dustynv/lerobot:r36.4.0`](https://hub.docker.com/r/dustynv/lerobot/tags) | `2024-12-13` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/lerobot:r36.4.0-20250305`](https://hub.docker.com/r/dustynv/lerobot/tags) | `2025-03-05` | `arm64` | `9.4GB` |
| &nbsp;&nbsp;[`dustynv/lerobot:r36.4.0-cu128`](https://hub.docker.com/r/dustynv/lerobot/tags) | `2025-02-09` | `arm64` | `8.5GB` |
| &nbsp;&nbsp;[`dustynv/lerobot:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/lerobot/tags) | `2025-04-24` | `arm64` | `7.0GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag lerobot)

# or explicitly specify one of the container images above
jetson-containers run dustynv/lerobot:r36.4.0-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/lerobot:r36.4.0-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag lerobot)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag lerobot) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build lerobot
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
