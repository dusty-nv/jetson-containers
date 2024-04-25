
* llama-index from https://www.llamaindex.ai/

### Starting llama-index container (only)

```bash
jetson-containers run $(./autotag llama-index)
```

### Running a starter RAG example with Ollama

This is based on the [official tutorial for local models](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/).

#### Data set up for the sample

On the Docker host console, copy the L4T-README text files to jetson-container's `/data` directory.

```bash
cd jetson-containers
mkdir -p data/documents/paul_grapham
wget "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1" -O data/documents/paul_grapham/paul_graham_essay.txt
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