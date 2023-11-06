
<a href="https://www.youtube.com/watch?v=9ObzbbBTbcc"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_llava_clip.gif"></a>

* Optimized LLM inference engine with support for AWQ and MLC quantization, multimodal agents, and live ASR/TTS.

## Text Chat

As an initial example, first test the console-based chat demo from [`__main__.py`](__main__.py)

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag local_llm) \
  python3 -m local_llm --api=mlc --model=meta-llama/Llama-2-7b-chat-hf
```
> For Llama-2 models, see [here](packages/llm/transformers/README.md#llama2) to request your access token from HuggingFace

The model will automatically be quantized the first time it's loaded (in this case, with MLC W4A16 quantization)

### Command-Line Options

Some of the noteworthy command-line options can be found in [`utils/args.py`](utils/args.py)

|                        |                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------|
| **Models**             |                                                                                           |
| `--model`              | The repo/name of the original unquantized model from HuggingFace Hub (or local path)      |
| `--quant`              | Either the API-specific quantization method to use, or path to quantized model            |
| `--api`                | The LLM model backend to use (`mlc, awq, auto_gptq, hf`)                                  |
| **Prompts**            |                                                                                           |
| `--prompt`             | Run this query (can be text, or a path to .txt file, and can be specified multiple times) |
| `--system-prompt`      | Sets the system instruction used at the beginning of the chat sequence.                   |
| `--chat-template`      | Manually set the chat template (`llama-2`, `llava-1`, `vicuna-v1`)                        |
| **Generation**         |                                                                                           |
| `--max-new-tokens`     | The maximum number of output tokens to generate for each response (default: 128)          |
| `--min-new-tokens`     | The minimum number of output tokens to generate (default: -1, disabled)                   |
| `--do-sample`          | Use token sampling during output with `--temperature` and `--top-p` settings              |
| `--temperature`        | Controls randomness of output with `--do-sample` (lower is less random, default: 0.7)     |
| `--top-p`              | Controls determinism/diversity of output with `--do-sample` (default: 0.95)               |
| `--repetition-penalty` | Applies a penalty for repetitive outputs (default: 1.0, disabled)                         |

### Automated Prompts

During testing, you can specify prompts on the command-line that will run sequentially:

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag local_llm) \
  python3 -m local_llm --api=mlc --model=meta-llama/Llama-2-7b-chat-hf \
    --prompt 'hi, how are you?' \
    --prompt 'whats the square root of 900?' \
    --prompt 'whats the previous answer times 4?' \
    --prompt 'can I get a recipie for french onion soup?'
```

### Multimodal (Llava)

If you load the Llava-1.5 model instead, you can enter image files into the prompt, followed by questions about them:

```bash
./run.sh $(./autotag local_llm) \
  python3 -m local_llm --api=mlc --model=liuhaotian/llava-v1.5-13b \
    --prompt '/data/images/fruit.jpg' \
    --prompt 'what kind of fruits do you see?' \
    --prompt '/data/images/dogs.jpg' \
    --prompt 'what breed of dogs are in the image?' \
    --prompt '/data/images/path.jpg' \
    --prompt 'what does the sign say?'
```

You can also enter `reset` (or `--prompt 'reset'`) to reset the chat history between images or responses.

## Voice Chat

<a href="https://www.youtube.com/watch?v=wzLHAgDxMjQ"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_70b_yt.jpg" width="800px"></a>

To enable the web UI and ASR/TTS for live conversations ...

