
<a href="https://www.youtube.com/watch?v=9ObzbbBTbcc"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_llava_clip.gif"></a>

* Optimized LLM inference engine with support for AWQ and MLC quantization, multimodal agents, and live ASR/TTS.

## Text Chat

As an initial example, first test the console-based chat demo from [`__main__.py`](__main__.py)

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag local_llm) \
  python3 -m local_llm --api=mlc --model=meta-llama/Llama-2-7b-chat-hf
```
> For Llama-2 models, see [here](/packages/llm/transformers/README.md#llama2) to request your access token from HuggingFace

The model will automatically be quantized the first time it's loaded (in this case, with MLC W4A16 quantization)

### Command-Line Options

Some of the noteworthy command-line options can be found in [`utils/args.py`](utils/args.py)

|                        |                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------|
| **Models**             |                                                                                           |
| `--model`              | The repo/name of the original unquantized model from HuggingFace Hub (or local path)      |
| `--quant`              | Either the API-specific quantization method to use, or path to quantized model            |
| `--api`                | The LLM model and quantization backend to use (`mlc, awq, auto_gptq, hf`)                 |
| **Prompts**            |                                                                                           |
| `--prompt`             | Run this query (can be text, or a path to .txt file, and can be specified multiple times) |
| `--system-prompt`      | Sets the system instruction used at the beginning of the chat sequence                    |
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

## Multimodal Chat

If you load a Llava vision-language model, you can enter image files into the prompt, followed by questions about them:

```bash
./run.sh $(./autotag local_llm) \
  python3 -m local_llm --api=mlc --model=liuhaotian/llava-v1.5-13b \
    --prompt '/data/images/fruit.jpg' \
    --prompt 'what kind of fruits do you see?' \
    --prompt 'reset' \
    --prompt '/data/images/dogs.jpg' \
    --prompt 'what breed of dogs are in the image?' \
    --prompt 'reset' \
    --prompt '/data/images/path.jpg' \
    --prompt 'what does the sign say?'
```

> [!WARNING]  
> Patch the model's [`config.json`](https://huggingface.co/liuhaotian/llava-v1.5-13b/blob/main/config.json) that was downloaded under `data/models/huggingface/models--liuhaotian--llava-v1.5-13b/snapshots/*`
>   * modify `"model_type": "llava",`
>   * to `"model_type": "llama",`
> Then re-run the command above - the quantization tools will then treat it like a Llama model (which it is)

Llava was trained to converse about one image at a time, hence the chat history is reset between images (otherwise the model tends to combine the features of all the images in the chat so far).  Multiple questions can be asked about each image though.

## Voice Chat

<a href="https://www.youtube.com/watch?v=wzLHAgDxMjQ"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_70b_yt.jpg" width="800px"></a>
> Interactive Voice Chat with Llama-2-70B on NVIDIA Jetson AGX Orin [`youtube.com/watch?v=wzLHAgDxMjQ`](https://www.youtube.com/watch?v=wzLHAgDxMjQ)

To enable the web UI and ASR/TTS for live conversations, follow the steps below.

### Start Riva Server

The ASR and TTS services use NVIDIA Riva with audio transformers and TensorRT.  The Riva server runs locally in it's own container.  Follow the steps from the [`riva-client:python`](/packages/riva-client) package to run and test the Riva server on your Jetson.

1. Start the Riva server on your Jetson by following [`riva_quickstart_arm64`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64)
2. Run some of the Riva ASR examples to confirm that ASR is working:  https://github.com/nvidia-riva/python-clients#asr
3. Run some of the Riva TTS examples to confirm that TTS is working:  https://github.com/nvidia-riva/python-clients#tts

You can also see this helpful video and guide from JetsonHacks for setting up Riva:  [**Speech AI on Jetson Tutorial**](https://jetsonhacks.com/2023/08/07/speech-ai-on-nvidia-jetson-tutorial/)

### Enabling HTTPS/SSL

Browsers require HTTPS to be used in order to access the client's microphone.  Hence, you'll need to create a self-signed SSL certificate and key:

```bash
$ cd /path/to/your/jetson-containers/data
$ openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj '/CN=localhost'
```

You'll want to place these in your [`jetson-containers/data`](/data) directory, because this gets automatically mounted into the containers under `/data`, and will keep your SSL certificate persistent across container runs.  When you first navigate your browser to a page that uses these self-signed certificates, it will issue you a warning since they don't originate from a trusted authority:

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/ssl_warning.jpg" width="400">

You can choose to override this, and it won't re-appear again until you change certificates or your device's hostname/IP changes.

### Start Web Agent

```bash
./run.sh \
  -e HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> \
  -e SSL_KEY=/data/key.pem \
  -e SSL_CERT=/data/cert.pem \
  $(./autotag local_llm) \
  python3 -m local_llm.agents.web_chat \
    --model meta-llama/Llama-2-7b-chat-hf \
    --api=mlc --verbose
```

You can then navigate your web browser to `https://HOSTNAME:8050` and unmute your microphone.

* The default port is 8050, but can be changed with `--web-port` (and `--ws-port` for the websocket port)
* To debug issues with client/server communication, use `--web-trace` to print incoming/outgoing websocket messages.
* During bot replies, the TTS model will pause output if you speak a few words in the mic to interrupt it.
* If you loaded a multimodal Llava model instead, you can drag-and-drop images from the client. 
  
## Tested Models

Llama 2:

* [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
* [`meta-llama/Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
* [`meta-llama/Llama-2-70b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)

LLaVA:  

* [`liuhaotian/llava-v1.5-7b`](https://huggingface.co/liuhaotian/llava-v1.5-7b)
* [`liuhaotian/llava-v1.5-13b`](https://huggingface.co/liuhaotian/llava-v1.5-13b)

Any fine-tuned version of Llama or Llava that shares the same architecture (or that is supported by the quantization API you have selected) should be compatible however, like Vicuna, CodeLlama, ect.  See [here](https://github.com/mlc-ai/mlc-llm/tree/main/mlc_llm/relax_model) for the MLC model architectures.
