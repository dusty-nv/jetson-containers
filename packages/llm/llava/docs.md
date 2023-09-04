
* LLaVa vision LLM from https://github.com/haotian-liu/LLaVA 
* Quantization is WIP :warning: (https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-gptq)

![hoover](/data/images/hoover.jpg)

### llava-llama-2-7b-chat

This is a LoRA applied to the original llama-2-7b-chat model, hence you need to request access and provide your HF token (or use [SaffalPoosh/llava-llama-2-7B-merged](https://huggingface.co/SaffalPoosh/llava-llama-2-7B-merged))

```bash
./run.sh --env HUGGING_FACE_HUB_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag llava) \
  python3 -m llava.serve.cli \
    --model-path liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview \
    --model-base meta-llama/Llama-2-7b-chat-hf \
    --image-file /data/images/hoover.jpg
```

```
USER: what does the road sign say?
ASSISTANT: The road sign says "Hoover Dam."
USER: how far away is the exit?
ASSISTANT: The exit is 1 mile away.
USER: what is the environment like?
ASSISTANT: The environment is desert-like, with a rocky landscape and a dirt road leading to the exit.
```

### llava-llama-2-13b-chat

```bash
./run.sh $(./autotag llava) \
  python3 -m llava.serve.cli \
    --model-path liuhaotian/llava-llama-2-13b-chat-lightning-preview \
    --image-file /data/images/hoover.jpg
```

```
USER: what does the text in the road sign say?
ASSISTANT: The text in the road sign says "Hoover Dam Exit 2 Mile."
USER: How far away is the exit?
ASSISTANT: The exit is two miles away from the current location.
USER: What kind of environment is it?
ASSISTANT: The environment is a desert setting, with a mountain in the background.
```
