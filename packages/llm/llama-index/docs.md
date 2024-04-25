
* llama-index from https://www.llamaindex.ai/

### Starting llama-index container (only)

```bash
jetson-containers run $(./autotag llama-index:samples)
```

### Running a sample with Ollama

#### Data set up for the sample

On the Docker host console, copy the L4T-README text files to jetson-container's `/data` directory.

```bash
cd jetson-containers
mkdir -p data/documents/L4T-README
cp /media/jetson/L4T-README/*.txt data/documents/L4T-README/
```

#### Docker-compose to run llama_index container with ollama container

> Here assumes we are on JetPack 6.0 DP and have followed the instruction [here](https://www.jetson-ai-lab.com/tips_ssd-docker.html#docker) for installing Docker.

Move to the `llama-index` package directory where `compose.yml` is saved, and use docker compose to run two containers.

```bash
cd ./packages/llm/llama-index
docker compose up
```

Open a new terminal and attach to the llama_index container.

```bash
docker exec -it llama-index bash
```

Once in the llama_index container, first download the Llama2 model using `ollama` command.

```bash
ollama pull llama2
```

Then, run the sample script to ask Jetson related questions (***"With USB device mode, what IP address Jetson gets? Which file should be edited in order to change the IP address assigned to Jetson?"***)to let the Llama-2 model answer based on the provided README files.

```bash
python3 samples/llamaindex_starter.py
```

It should answer something like this;

```text
Based on the context provided, the static IP address assigned to Jetson is 192.168.55.100. To change the IP address assigned to Jetson, you should edit the "Property" section of the "Remote NDIS Compatible Device" interface in the Network Connections settings on your host system. The file that should be edited is /opt/nvidia/l4t-usb-device-mode/nv-l4t-usb-device-mode-config.sh on Jetson.
```

> The correct answer to the first question is `192.168.55.1`.