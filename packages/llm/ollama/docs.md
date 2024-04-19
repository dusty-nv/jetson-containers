
* Ollama from https://github.com/ollama/ollama with CUDA enabled (found under `/bin/ollama`)
* Thanks to [`@remy415`](https://github.com/remy415) for getting Ollama working on Jetson and contributing the Dockerfile ([PR #465](https://github.com/dusty-nv/jetson-containers/pull/465))

## Ollama Server

Run the local Ollama server instance as a daemon in the background, either of these ways:

```
# models cached under jetson-containers/data
jetson-containers run --detach --name ollama $(autotag ollama)

# models cached under your user's home directory
docker run --runtime nvidia -it --rm --network=host -v ~/ollama:/root/.ollama dustynv/ollama:r36.2.0
```

Ollama stores models in `/usr/share/ollama/.ollama/models` by default. Create a symlink on the host to sync containers to any binaries run straight from the console.  Ensure the folder doesn't exist, and if it does move its content first then remove the folder:

```
sudo mv /usr/share/ollama/.ollama/* $(jetson-containers data)/models/ollama
sudo rm -r /usr/share/ollama/.ollama
sudo ln -s $(jetson-containers data)/models/ollama /usr/share/ollama/.ollama
```

## Ollama Client

Start the Ollama CLI front-end with your desired model (for example: mistral 7b)

```
jetson-containers run $(autotag ollama) /bin/ollama run mistral
```

<img src="https://github.com/dusty-nv/jetson-containers/blob/docs/docs/images/ollama_cli.gif?raw=true" width="750px"></img>

Or you can install Ollama's released binaries for arm64 (without CUDA, which only the server needs)

```
# download the latest ollama release for arm64 into /bin
sudo wget https://github.com/ollama/ollama/releases/download/$(git ls-remote --refs --sort="version:refname" --tags https://github.com/ollama/ollama | cut -d/ -f3- | sed 's/-rc.*//g' | tail -n1)/ollama-linux-arm64 -O /bin/ollama
sudo chmod +x /bin/ollama

# use the client like normal outside container
/bin/ollama run mistral
```

## Open WebUI

To run [Open WebUI](https://github.com/open-webui/open-webui) server for client browsers to connect to, use the `open-webui` container:

```
docker run -it --rm --network=host --add-host=host.docker.internal:host-gateway ghcr.io/open-webui/open-webui:main
```

You can then navigate your browser to `http://JETSON_IP:8080`, and create a fake account to login (these credentials are only stored locally)

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/ollama_open_webui.jpg" width="800px"></img>

## Memory Usage

| Model                                                                           |          Quantization         | Memory (MB) |
|---------------------------------------------------------------------------------|:-----------------------------:|:-----------:|
| [`TheBloke/Llama-2-7B-GGUF`](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)   |  `llama-2-7b.Q4_K_S.gguf`     |    5,268    |
| [`TheBloke/Llama-2-13B-GGUF`](https://huggingface.co/TheBloke/Llama-2-13B-GGUF) | `llama-2-13b.Q4_K_S.gguf`     |    8,609    |
| [`TheBloke/LLaMA-30b-GGUF`](https://huggingface.co/TheBloke/LLaMA-30b-GGUF)     | `llama-30b.Q4_K_S.gguf`       |    19,045   |
| [`TheBloke/Llama-2-70B-GGUF`](https://huggingface.co/TheBloke/Llama-2-70B-GGUF) | `llama-2-70b.Q4_K_S.gguf`     |    37,655   |
