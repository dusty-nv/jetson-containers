# text-generation-webui

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


![](https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/images/text-generation-webui_sf-trip.gif)

* text-generation-webui from https://github.com/oobabooga/text-generation-webui (found under `/opt/text-generation-webui`)
* includes CUDA-optimized model loaders for: [`llama.cpp`](/packages/llm/llama_cpp) [`exllama`](/packages/llm/exllama) [`AutoGPTQ`](/packages/llm/auto_gptq) [`transformers`](/packages/llm/transformers)
* see the tutorial on the [**Jetson Generative AI Playground**](https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/tutorial_text-generation.html)

This container has a default run command that will automatically start the webserver like this:

```bash
cd /opt/text-generation-webui && python3 server.py \
  --model-dir=/data/models/text-generation-webui \
  --listen --verbose
```

To launch the container, run the command below, and then navigate your browser to `http://HOSTNAME:7860`

```bash
./run.sh $(./autotag text-generation-webui)
```

### Command-Line Options

While the server and models are dynamically configurable from within the webui at runtime, see here for the list of command-line options that you can set at start-up:   

* https://github.com/oobabooga/text-generation-webui/tree/main#basic-settings

For example, after you've [downloaded a model](#downloading-models), you can load it directly at startup like so:

```bash
./run.sh $(./autotag text-generation-webui) /bin/bash -c \
  "cd /opt/text-generation-webui && python3 server.py \
	--model-dir=/data/models/text-generation-webui \
	--model=llama-2-13b-chat.ggmlv3.q4_0.bin \
	--loader=llamacpp \
	--n-gpu-layers=128 \
	--listen --chat --verbose
```

### Downloading Models 

See [here](https://github.com/oobabooga/text-generation-webui/tree/main#downloading-models) for instructions for downloading models - you can do this from within the webui, or by running their [`download-model.py`](https://github.com/oobabooga/text-generation-webui/blob/main/download-model.py) script in the container:

```bash
./run.sh --workdir=/opt/text-generation-webui $(./autotag text-generation-webui) /bin/bash -c \
  'python3 download-model.py --output=/data/models/text-generation-webui TheBloke/Llama-2-7b-Chat-GPTQ'
```

This will download specified model from [HuggingFace Hub](https://huggingface.co/models) and place it under the `/data/models/text-generation-webui` mounted directory (which is where you should store models so they aren't lost when the container exits)

### Tips and Tricks

* The fastest model loader to use is currently [llama.cpp](/packages/llm/llama_cpp) with 4-bit quantized GGML models
  * Remember to set `n-gpu-layers` to 128 in the loader settings
  * If you're using Llama-2-70B, set `n_gqa` to 8 (otherwise an error will occur)
  * Tested using the `*q4_0.bin` model quantizations
* Unless you loaded a model fine-tuned for chat, use text completion mode in the `Default` or `Notebook` tab
* If you're using a Llama-2 chat model, use the `Instruct` chat mode and set the Instruction Template to `Llama-v2` (in the `Parameters` tab)
  * This will make sure the correct [chat prompt format](https://huggingface.co/blog/llama2#how-to-prompt-llama-2) is being used for Llama-2
  * You can alter your agent's behavior/personality/ect by tweaking this prompt template (the `Answer the questions.` part)

### Things to do with your LLM

```
What games do you like to play?
```

```
I'm a large language model, so I can play text-based games and answer questions on a wide variety of topics. Here are some game ideas:

- 20 Questions:  You or I can think of an object, and the other player tries to guess what it is by asking up to 20 yes-or-no questions.  The goal is to guess the object in as few questions as possible.
- Hangman: I can think of a word or phrase and you try to guess it by suggesting letters. Each incorrect guess will bring the hangman one step closer to being complete!
- Word association: I'll give you a word and you have to come up with a word that is related to it. We can keep playing back and forth as long as you'd like!
- Trivia: I can provide questions on a wide range of topics, from history to science to pop culture. See how many questions you can answer correctly!
- Storytelling: I can start telling a story and then stop at a cliffhanger. You can then try to guess what happens next or even take over the storytelling and continue it in your own
- Jokes: I love to tell jokes and make people laugh with my "Dad humor"!  Knock knock!  *giggles*
```


<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`text-generation-webui`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`text-generation-webui_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/text-generation-webui_jp51.yml?label=text-generation-webui:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/text-generation-webui_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`bitsandbytes`](/packages/llm/bitsandbytes) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/rust) [`transformers`](/packages/llm/transformers) [`auto_gptq`](/packages/llm/auto_gptq) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`exllama`](/packages/llm/exllama) [`llama_cpp`](/packages/llm/llama_cpp) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-text-generation`](/packages/l4t/l4t-text-generation) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/text-generation-webui:r35.2.1`](https://hub.docker.com/r/dustynv/text-generation-webui/tags) `(2023-08-17, 6.2GB)`<br>[`dustynv/text-generation-webui:r35.3.1`](https://hub.docker.com/r/dustynv/text-generation-webui/tags) `(2023-08-15, 6.2GB)`<br>[`dustynv/text-generation-webui:r35.4.1`](https://hub.docker.com/r/dustynv/text-generation-webui/tags) `(2023-08-13, 6.2GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/text-generation-webui:r35.2.1`](https://hub.docker.com/r/dustynv/text-generation-webui/tags) | `2023-08-17` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/text-generation-webui:r35.3.1`](https://hub.docker.com/r/dustynv/text-generation-webui/tags) | `2023-08-15` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/text-generation-webui:r35.4.1`](https://hub.docker.com/r/dustynv/text-generation-webui/tags) | `2023-08-13` | `arm64` | `6.2GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag text-generation-webui)

# or explicitly specify one of the container images above
./run.sh dustynv/text-generation-webui:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/text-generation-webui:r35.2.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag text-generation-webui)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag text-generation-webui) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh text-generation-webui
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
