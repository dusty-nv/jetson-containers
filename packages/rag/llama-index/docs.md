
* llama-index from https://www.llamaindex.ai/

### Starting `llamaindex` container

```bash
jetson-containers run $(./autotag llama-index)
```

This will start the `ollama` server as well as Jupyter Lab server inside the container.

### Running a starter RAG example with Ollama

This is based on the [official tutorial for local models](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/).

#### Jupyter notebook version

When you run start the `llamaindex` container, you should see lines like this on the terminal.

```
JupyterLab URL:   http://10.110.50.241:8888 (password "nvidia")
JupyterLab logs:  /data/logs/jupyter.log
```

On your Jetson desktop GUI, or on a PC on the same network as Jetson, open your web browser and access the address.

When prompted, type the password `nvidia` and log in.

Jupyter Lab UI should show up, with `LlamaIndex_Local-Models.ipynb` listed in the left navigator pane.

Select and open the `LlamaIndex_Local-Models.ipynb`

Follow the guide on the Jupyter notebook.

####  Python script version

After starting the `llamaindex` container, you should be on `root@<hostname>` console.

First, download the Llama2 model using `ollama` command.

```bash
ollama pull llama2
```

> This downloads the default 7-billion parameter Llama2 model.
> You can optionally specify `ollma2:13b` and `ollma2:70b` for other variations, and change the Python script (line 13) accordingly.

Type the following to start the sample Python script.

```bash
python3 /opt/llama-index/llamaindex_starter.py
```

