
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



