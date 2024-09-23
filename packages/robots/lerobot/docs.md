
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
<<<<<<< HEAD
jetson-containers run --shm-size=4g -w /opt/lerobot $(autotag lerobot) \
  python3 lerobot/scripts/visualize_dataset.py \
=======
jetson-containers run --shm-size=4g $(autotag lerobot) python3 /opt/lerobot/lerobot/scripts/visualize_dataset.py \
>>>>>>> 59917b7d5e35ff9c262a478186a809f368152f02
    --repo-id lerobot/pusht \
    --episode-index 0
```

### Example: Evaluate a pretrained policy

See the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#evaluate-a-pretrained-policy).

```bash
<<<<<<< HEAD
jetson-containers run --shm-size=4g -w /opt/lerobot $(autotag lerobot) \
  python3 lerobot/scripts/eval.py \
=======
jetson-containers run --shm-size=4g $(autotag lerobot) python3 /opt/lerobot/lerobot/scripts/eval.py \
>>>>>>> 59917b7d5e35ff9c262a478186a809f368152f02
    -p lerobot/diffusion_pusht \
    eval.n_episodes=10 \
    eval.batch_size=10
```

### Example: Train your own policy

See the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#train-your-own-policy).

```bash
<<<<<<< HEAD
jetson-containers run --shm-size=4g -w /opt/lerobot $(autotag lerobot) \
  python3 lerobot/scripts/train.py \
=======
jetson-containers run --shm-size=4g $(autotag lerobot) python3 /opt/lerobot/lerobot/scripts/train.py \
>>>>>>> 59917b7d5e35ff9c262a478186a809f368152f02
    policy=act \
    env=aloha \
    env.task=AlohaInsertion-v0 \
    dataset_repo_id=lerobot/aloha_sim_insertion_human 
```
