## Distributed inference on Jetson devices

### Available for NCCL v2.27.7 and L4T version 36.x

Experimental support for distributed inference on Jetson devices that use NCCL v2.27.7 was added in this [PR](https://github.com/dusty-nv/jetson-containers/pull/1429).

Supports Multi-node Parallelism using tensor parallel, as explained in[ vLLM's Multi-node deployment docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html#multi-node-deployment)

### How to use:
Build `PyTorch` from source with NCCL support enabled like this:

```bash

ENABLE_NCCL_DISTRIBUTED_JETSON=1 \
PYTORCH_FORCE_BUILD=on \
jetson-containers build pytorch
```
Or if building `vLLM`:

```bash

ENABLE_NCCL_DISTRIBUTED_JETSON=1 \
PYTORCH_FORCE_BUILD=on \
jetson-containers build vllm
```

#### This will:
- build `NCCL` with the patch that adds support for Jetson
- build `PyTorch` with `USE_SYSTEM_NCCL` enabled


### Running on 2 Jetson devices:

#### Example with 2x `Jetson Orin AGX 64 GB RAM` dev kits
Below there are 2 ways to run distributed inference on 2 Jetson devices:
1. Manual docker method: running docker containers and starting Ray on both devices (requires docker knowledge)
2. Using the helper script at `./distributed-inference.sh` (easier), just skip the `Manual docker method` below if using this method

- In this example one Jetson is referenced as `HEAD NODE` and the other as `WORKER NODE`
- Tested with images:
  - `mitakad/vllm:0.12.0-r36.4.tegra-aarch64-cp310-cu126-22.04-truncated`
  - `mitakad/vllm:0.11.2-r36.4.tegra-aarch64-cp310-cu126-22.04-truncated`

#### 1. Manual docker method:
- On the `HEAD NODE`, first get the IP address of the device and the network interface name to be used for communication (e.g. `eno1`, `eth0`, etc):
  - `VLLM_HOST_IP` is the IP address of the `HEAD NODE`
  - `VLLM_IFNAME` is the network interface name to be used for communication on `HEAD NODE`
  - `HF_TOKEN` is your Hugging Face token if you are using models from Hugging Face Hub
  - `JETSON_CONTAINERS_PATH` is the path where jetson-containers repo is stored on `HEAD NODE`
  - Then run the container with the following command, first replacing the `...` with the appropriate values:
```bash
VLLM_HOST_IP=... && \
VLLM_IFNAME=... && \
HF_TOKEN=... && \
docker run --runtime nvidia -it \
--network host \
--shm-size=8g \
--privileged \
--ipc=host \
-v /dev/shm:/dev/shm \
-v $(jetson-containers data):/data \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e VLLM_HOST_IP=${VLLM_HOST_IP} \
-e UCX_NET_DEVICES=${VLLM_IFNAME} \
-e NCCL_SOCKET_IFNAME=${VLLM_IFNAME} \
-e GLOO_SOCKET_IFNAME=${VLLM_IFNAME} \
-e TP_SOCKET_IFNAME=${VLLM_IFNAME} \
-e TIKTOKEN_ENCODINGS_BASE=/data/encodings \
-e HF_TOKEN=${HF_TOKEN} \
--name vllm_112_eth \
mitakad/vllm:0.11.2-r36.4.tegra-aarch64-cp310-cu126-22.04-truncated
```
- On the `WORKER NODE` run same command as above but this time set the variables accordingly:
  - `VLLM_HOST_IP` is the IP address of the `WORKER NODE`
  - `VLLM_IFNAME` is the network interface name to be used for communication on `WORKER NODE`
  - `HF_TOKEN` is your Hugging Face token if you are using models from Hugging Face Hub
  - `JETSON_CONTAINERS_PATH` is the path where jetson-containers repo is stored on `WORKER NODE`

- Next we need to start Ray on both devices ([What is Ray?](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/#what-is-ray))
  - On the `HEAD NODE` container run (replace values in `{{}}`):
```bash
ray start -v --block --head --port 6379 --node-ip-address {{ set HEAD NODE IP here}} --num-gpus 1
```
  - On the `WORKER NODE` container run (replace values in `{{}}`):
```bash
ray start -v --block --address {{ set HEAD NODE IP here }}:6379 --node-ip-address {{ set WORKER NODE IP here }} --num-gpus 1
```

- Finally, on the `HEAD NODE` container run the following command to start distributed inference (replace values in `{{}}`):
```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --port 8000 \
    --served-model-name Qwen3-4B-Instruct-2507-FP8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.5 \
    --max-num-seqs 10 \
    --tensor-parallel-size 2
```

- You can now send requests to the `HEAD NODE` IP address on port `8000` to perform distributed inference across both Jetson devices:
- Example using `curl`:
```bash
curl --location 'http://{{ set HEAD NODE IP here}}:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
            "model": "Qwen3-4B-Instruct-2507-FP8",
            "messages": [
                {
                    "role": "user",
                    "content": "Avast! What'\''s the future of AI?"
                }
            ]
        }'
```

#### 2. Helper script to run distributed inference on 2 Jetson devices at `./distributed-inference.sh`

Below is an example of how to use the helper script.

For `HEAD NODE` IP address we use `192.168.1.100`.

For `WORKER NODE` IP address we use `192.168.1.101`.

Using network interface `eno1` for communication.

- On `HEAD NODE`:
    1. This is the command <u>without</u> the example values filled in:
    ```bash
        bash distributed-inference.sh \
             mitakad/vllm:0.11.2-r36.4.tegra-aarch64-cp310-cu126-22.04-truncated \
             <HEAD NODE IP> \
             --head  \
             <network_interface> \
             -e VLLM_HOST_IP=<HEAD NODE IP> \
             -e HF_TOKEN=<huggingface_token>
    ```

    2. This is the command <u>with</u> the example values filled in:

    ```bash
    bash distributed-inference.sh \
        mitakad/vllm:0.11.2-r36.4.tegra-aarch64-cp310-cu126-22.04-truncated \
        192.168.1.100 \
        --head \
        eno1 \
        -e VLLM_HOST_IP=192.168.1.100 \
        -e HF_TOKEN=hf_xxx
    ```

- On `WORKER NODE`:

    1. This is the command <u>without</u> the example values filled in:
    ```bash
        bash distributed-inference.sh \
             mitakad/vllm:0.11.2-r36.4.tegra-aarch64-cp310-cu126-22.04-truncated \
             <HEAD NODE IP> \
             --worker  \
             <network_interface> \
             -e VLLM_HOST_IP=<WORKER NODE IP> \
             -e HF_TOKEN=<huggingface_token>
    ```

    2. This is the command <u>with</u> the example values filled in:
    ```bash
    bash distributed-inference.sh \
        mitakad/vllm:0.11.2-r36.4.tegra-aarch64-cp310-cu126-22.04-truncated \
        192.168.1.100 \
        --worker \
        eno1 \
        -e VLLM_HOST_IP=192.168.1.101 \
        -e HF_TOKEN=hf_xxx
    ```

- Then start the distributed inference on the `HEAD NODE` container:
  1. example command for `gpt-oss-20b` (~ 12 t/s on 10 Gbit connection):

    ```bash
    docker exec -it jetson-cluster-node \
    vllm serve openai/gpt-oss-20b --port 8000 \
        --served-model-name gpt-oss-20b \
        --max-model-len 2048 \
        --max-num-batched-tokens 2048 \
        --gpu-memory-utilization 0.5 \
        --kv-cache-memory-bytes 3000M \
        --max-num-seqs 10 \
        --tensor-parallel-size 2 \
        --enable-expert-parallel \
        --distributed-executor-backend ray
    ```

  2. example command for `gpt-oss-120b` (~ 9 t/s on 10 Gbit connection):

    ```bash
    docker exec -it jetson-cluster-node \
    vllm serve openai/gpt-oss-120b --port 8000 \
        --served-model-name gpt-oss-120b \
        --max-model-len 2048 \
        --max-num-batched-tokens 2048 \
        --kv-cache-memory-bytes 3000M \
        --max-num-seqs 10 \
        --tensor-parallel-size 2 \
        --enable-expert-parallel \
        --distributed-executor-backend ray
    ```
  3. example command for `QuantTrio/GLM-4.5-Air-AWQ-FP16Mix` (~ 6 t/s on 10 Gbit connection):

    ```bash
    docker exec -it jetson-cluster-node \
    vllm serve QuantTrio/GLM-4.5-Air-AWQ-FP16Mix --port 8000 \
        --served-model-name GLM-4.5-Air-AWQ-FP16Mix \
        --max-model-len 2048 \
        --max-num-batched-tokens 2048 \
        --kv-cache-memory-bytes 6000M \
        --max-num-seqs 10 \
        --tensor-parallel-size 2 \
        --enable-expert-parallel \
        --distributed-executor-backend ray
    ```
  
- Test with `curl` (replace values in `{{}}`):
```bash
curl --location 'http://{{ set HEAD NODE IP here}}:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
            "model": "{{ served-model-name }}",
            "messages": [
                {
                    "role": "user",
                    "content": "Why is the sky blue?"
                }
            ]
        }'
```

### Notes:
The existing ethernet connection speed between the Jetson devices will impact performance. 
Faster connection will provide better performance.
You could use PCIe ports to connect NICs to get better connection speeds (over 10 Gbit/s).
