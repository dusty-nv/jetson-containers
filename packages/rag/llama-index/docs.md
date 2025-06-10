
* llama-index from https://www.llamaindex.ai/

## Starting `llamaindex` container

```bash
jetson-containers run $(autotag llama-index:samples)
```

This will start the `ollama` server as well as Jupyter Lab server inside the container.

## Running a RAG example with Ollama

This is based on the [official tutorial for local models](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/).

#### Jupyter Notebook Version

When you run start the `llama-index` container, you should see lines like this on the terminal.

```
JupyterLab URL:   http://192.168.1.10:8888 (password "nvidia")
JupyterLab logs:  /data/logs/jupyter.log
```

On your Jetson desktop GUI, or on a PC on the same network as Jetson, open your web browser and access the address. When prompted, type the password `nvidia` and log in.

Jupyter Lab UI should show up, with [`LlamaIndex_Local-Models.ipynb`](samples/LlamaIndex_Local-Models.ipynb) listed in the left navigator pane - open it, and follow the guide in the Jupyter notebook.

####  Python Version

After starting the `llamaindex` container, you should be on `root@<hostname>` console. First, download the Llama2 model using `ollama`

```bash
ollama pull llama2
```

This downloads the default 7-billion parameter Llama2 model - you can optionally specify `ollma2:13b` and `ollma2:70b` for other variations, and change the Python script (line 13) accordingly. Then type the following to start the sample Python script:

```bash
python3 /opt/llama-index/llamaindex_starter.py
```

