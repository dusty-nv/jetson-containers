
<img src="https://github.com/user-attachments/assets/6a8967e1-f9dd-463f-906b-d9fd1f44450f">

* lerobot project from https://github.com/huggingface/lerobot/
* see the tutorial at [**Jetson AI Lab**](https://jetson-ai-lab.com/tutorial_lerobot.html) (Coming scoon)

## Usage Examples

### Example: Visualize datasets

*Check the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#visualize-datasets).*

On Docker host (Jetson native), first launch rerun.io.

```bash
pip install rerun-sdk
rerun
```

Then, start the docker container to run the visualization script.

```bash
jetson-containers run $(autotag lerobot) python3 lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

### Example: Evaluate a pretrained policy

*Check the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#evaluate-a-pretrained-policy).*

```bash
jetson-containers run $(autotag lerobot) python3 lerobot/scripts/eval.py \
    -p lerobot/diffusion_pusht \
    eval.n_episodes=10 \
    eval.batch_size=10
```

### Example: Train your own policy

*Check the [original instruction on lerobot repo](https://github.com/huggingface/lerobot/?tab=readme-ov-file#train-your-own-policy).*

```bash
jetson-containers run --shm-size=6g $(autotag lerobot) python3 lerobot/scripts/train.py \
    policy=act \
    env=aloha \
    env.task=AlohaInsertion-v0 \
    dataset_repo_id=lerobot/aloha_sim_insertion_human 
```
