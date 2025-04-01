
![](https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/images/text-generation-webui_sf-trip.gif)

* text-generation-webui from https://github.com/oobabooga/text-generation-webui (found under `/opt/text-generation-webui`)
* includes CUDA-optimized model loaders for: [`llama.cpp`](/packages/llm/llama_cpp) [`exllama2`](/packages/llm/exllama) [`AutoGPTQ`](/packages/llm/auto_gptq) [`transformers`](/packages/llm/transformers)
* see the tutorial at the [**Jetson Generative AI Lab**](https://www.jetson-ai-lab.com/tutorial_text-generation.html)

> [!WARNING]  
> If you're using the llama.cpp loader, the model format has changed from GGML to GGUF.  Existing GGML models can be converted using the `convert-llama-ggmlv3-to-gguf.py` script in [`llama.cpp`](https://github.com/ggerganov/llama.cpp) (or you can often find the GGUF conversions on [HuggingFace Hub](https://huggingface.co/models?search=GGUF))

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

While the server and models are dynamically configurable from within the webui at runtime, see here for optional command-line settings:

* https://github.com/oobabooga/text-generation-webui/tree/main#basic-settings

For example, after you've [downloaded a model](#downloading-models), you can load it directly at startup like so:

```bash
./run.sh $(./autotag text-generation-webui) /bin/bash -c \
  "cd /opt/text-generation-webui && python3 server.py \
	--model-dir=/data/models/text-generation-webui \
	--model=llama-2-13b-chat.Q4_K_M.gguf \
	--loader=llamacpp \
	--n-gpu-layers=128 \
	--listen --chat --verbose
```

### Downloading Models 

See [here](https://github.com/oobabooga/text-generation-webui/tree/main#downloading-models) for instructions for downloading models - you can do this from within the webui, or by running their [`download-model.py`](https://github.com/oobabooga/text-generation-webui/blob/main/download-model.py) script:

```bash
./run.sh --workdir=/opt/text-generation-webui $(./autotag text-generation-webui) /bin/bash -c \
  'python3 download-model.py --output=/data/models/text-generation-webui TheBloke/Llama-2-7b-Chat-GPTQ'
```

This will download specified model from [HuggingFace Hub](https://huggingface.co/models) and place it under the `/data/models/text-generation-webui` mounted directory (which is where you should store models so they aren't lost when the container exits)

### Tips and Tricks

* The fastest model loader to use is currently [llama.cpp](/packages/llm/llama_cpp) with 4-bit quantized GGUF models
  * Remember to set `n-gpu-layers` to 128 in the loader settings
  * If you're using Llama-2-70B, set `n_gqa` to 8 (otherwise an error will occur)
  * Tested using the recommended `Q4_K_M` model quantizations
* Unless you loaded a model fine-tuned for chat, use text completion mode in the `Default` or `Notebook` tab
* If you're using a Llama-2 chat model, use the `Instruct` chat mode and set the Instruction Template to `Llama-v2` (in the `Parameters` tab)
  * This will make sure the correct [chat prompt format](https://huggingface.co/blog/llama2#how-to-prompt-llama-2) is being used for Llama-2
  * You can alter your agent's behavior/personality/ect by tweaking this prompt template (the `Answer the questions.` part)

### Things to do with your LLM

```
>> What games do you like to play?

I'm a large language model, so I can play text-based games and answer questions on a wide variety of topics. Here are some game ideas:

- 20 Questions:  You or I can think of an object, and the other player tries to guess what it is by asking up to 20 yes-or-no questions.  The goal is to guess the object in as few questions as possible.
- Hangman: I can think of a word or phrase and you try to guess it by suggesting letters. Each incorrect guess will bring the hangman one step closer to being complete!
- Word association: I'll give you a word and you have to come up with a word that is related to it. We can keep playing back and forth as long as you'd like!
- Trivia: I can provide questions on a wide range of topics, from history to science to pop culture. See how many questions you can answer correctly!
- Storytelling: I can start telling a story and then stop at a cliffhanger. You can then try to guess what happens next or even take over the storytelling and continue it in your own
- Jokes: I love to tell jokes and make people laugh with my "Dad humor"!  Knock knock!  *giggles*
```

