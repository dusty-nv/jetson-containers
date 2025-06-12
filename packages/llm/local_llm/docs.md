
> [!NOTE]  
> This source outgrew being hosted within jetson-containers and moved to:
> * Repo - [`github.com/dusty-nv/NanoLLM`](https://github.com/dusty-nv/NanoLLM)
> * Docs - [`dusty-nv.github.io/NanoLLM`](https://dusty-nv.github.io/NanoLLM)
> * Jetson AI Lab - [Live Llava](https://www.jetson-ai-lab.com/tutorial_live-llava.html), [NanoVLM](https://www.jetson-ai-lab.com/tutorial_nano-vlm.html), [SLM](https://www.jetson-ai-lab.com/tutorial_slm.html)
>
> It will remain here for backwards compatability, but future updates will be made to NanoLLM.

<details>
<summary><b>ARCHIVED DOCUMENTATION</b></summary>
<br/>
<a href="https://www.youtube.com/watch?v=9ObzbbBTbcc"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_llava_clip.gif"></a>

* Optimized LLM inference engine with support for AWQ and MLC quantization, multimodal agents, and live ASR/TTS.
* Web UI server using Flask, WebSockets, WebAudio, HTML5, Bootstrap5.
* Modes to run: [Text Chat](#text-chat), [Multimodal Chat](#multimodal-chat), [Voice Chat](#voice-chat), [Live Llava](#live-llava)

> [!NOTE]  
> Tested models:
>   * [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
>   * [`meta-llama/Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
>   * [`meta-llama/Llama-2-70b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
>
> Small Language Models ([SLMs](https://www.jetson-ai-lab.com/tutorial_slm.html))
>   * [`stabilityai/stablelm-2-zephyr-1_6b`](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)
>   * [`stabilityai/stablelm-zephyr-3b`](https://huggingface.co/stabilityai/stablelm-zephyr-3b)
>   * [`NousResearch/Nous-Capybara-3B-V1.9`](https://huggingface.co/NousResearch/Nous-Capybara-3B-V1.9)
>   * [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
>   * [`princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT`](https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT)
>   * [`google/gemma-2b-it`](https://huggingface.co/google/gemma-2b-it)
>   * [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2)
>
> Vision Language Models ([VLMs](https://www.jetson-ai-lab.com/tutorial_llava.html))
>   * [`liuhaotian/llava-v1.5-7b`](https://huggingface.co/liuhaotian/llava-v1.5-7b)
>   * [`liuhaotian/llava-v1.5-13b`](https://huggingface.co/liuhaotian/llava-v1.5-13b)
>   * [`liuhaotian/llava-v1.6-vicuna-7b`](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b)
>   * [`liuhaotian/llava-v1.6-vicuna-13b`](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b)
>   * [`Efficient-Large-Model/VILA-2.7b`](https://huggingface.co/Efficient-Large-Model/VILA-2.7b)
>   * [`Efficient-Large-Model/VILA-7b`](https://huggingface.co/Efficient-Large-Model/VILA-7b)
>   * [`Efficient-Large-Model/VILA-13b`](https://huggingface.co/Efficient-Large-Model/VILA-13b)
>   * [`NousResearch/Obsidian-3B-V0.5`](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
>
> For Llama-2 models, see [here](/packages/llm/transformers/README.md#llama2) to request your access token from HuggingFace.

## Text Chat

As an initial example, first test the console-based chat demo from [`chat/__main__.py`](chat/__main__.py)

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> \
  $(./autotag local_llm) \
    python3 -m local_llm.chat --api=mlc \
      --model=meta-llama/Llama-2-7b-chat-hf
```

The model will automatically be quantized the first time it's loaded (in this case, with MLC and 4-bit).  Other fine-tuned versions of Llama that have the same architecture (or are supported by the quantization API you have selected) should be compatible - see [here](https://github.com/mlc-ai/mlc-llm/tree/main/mlc_llm/relax_model) for MLC.

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
| `--max-context-len`    | The maximum chat history context length (in tokens), lower to reduce memory usage         |
| `--max-new-tokens`     | The maximum number of output tokens to generate for each response (default: 128)          |
| `--min-new-tokens`     | The minimum number of output tokens to generate (default: -1, disabled)                   |
| `--do-sample`          | Use token sampling during output with `--temperature` and `--top-p` settings              |
| `--temperature`        | Controls randomness of output with `--do-sample` (lower is less random, default: 0.7)     |
| `--top-p`              | Controls determinism/diversity of output with `--do-sample` (default: 0.95)               |
| `--repetition-penalty` | Applies a penalty for repetitive outputs (default: 1.0, disabled)                         |

### Automated Prompts

During testing, you can specify prompts on the command-line that will run sequentially:

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> \
  $(./autotag local_llm) \
    python3 -m local_llm.chat --api=mlc \
      --model=meta-llama/Llama-2-7b-chat-hf \
      --prompt 'hi, how are you?' \
      --prompt 'whats the square root of 900?' \
      --prompt 'whats the previous answer times 4?' \
      --prompt 'can I get a recipie for french onion soup?'
```

You can also load JSON files containing prompt sequences, like from [`/data/prompts/qa.json`](/data/prompts/qa.json)

## Multimodal Chat

If you load a Llava vision-language model, you can enter image files into the prompt, followed by questions about them:

```bash
./run.sh $(./autotag local_llm) \
  python3 -m local_llm.chat --api=mlc \
    --model=liuhaotian/llava-v1.5-13b \
    --prompt '/data/images/fruit.jpg' \
    --prompt 'what kind of fruits do you see?' \
    --prompt 'reset' \
    --prompt '/data/images/dogs.jpg' \
    --prompt 'what breed of dogs are in the image?' \
    --prompt 'reset' \
    --prompt '/data/images/path.jpg' \
    --prompt 'what does the sign say?'
```

Llava was trained to converse about one image at a time, hence the chat history is reset between images (otherwise the model tends to combine the features of all the images in the chat so far).  Multiple questions can be asked about each image though.

By omitting `--prompt`, you can chat interactively from the terminal.  If you enter an image filename, it will load that image, and then asking you for the prompt.  Entering `clear` or `reset` will reset the chat history. 

## Voice Chat

<a href="https://www.youtube.com/watch?v=wzLHAgDxMjQ"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_70b_yt.jpg" width="800px"></a>
> [Interactive Voice Chat with Llama-2-70B on NVIDIA Jetson AGX Orin](https://www.youtube.com/watch?v=wzLHAgDxMjQ)

To enable the web UI and ASR/TTS for live conversations, follow the steps below.

### Start Riva Server

The ASR and TTS services use NVIDIA Riva with audio transformers and TensorRT.  The Riva server runs locally in it's own container.  Follow the steps from the [`riva-client:python`](/packages/audio/riva-client) package to run and test the Riva server on your Jetson.

1. Start the Riva server on your Jetson by following [`riva_quickstart_arm64`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64)
2. Run some of the Riva ASR examples to confirm that ASR is working:  https://github.com/nvidia-riva/python-clients#asr
3. Run some of the Riva TTS examples to confirm that TTS is working:  https://github.com/nvidia-riva/python-clients#tts

You can also see this helpful video and guide from JetsonHacks for setting up Riva:  [**Speech AI on Jetson Tutorial**](https://jetsonhacks.com/2023/08/07/speech-ai-on-nvidia-jetson-tutorial/)

### Enabling HTTPS/SSL

Browsers require HTTPS in order to access the client's microphone.  A self-signed SSL certificate was built into the container like this:

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj '/CN=localhost'
```

The container's certificate is found under `/etc/ssl/private` and is automatically used, so HTTPS/SSL is enabled by default for these web UI's (you can change the PEM certificate/key used by setting the `SSL_KEY` and `SSL_CERT` environment variables).  When you first navigate your browser to a page that uses these self-signed certificates, it will issue you a warning since they don't originate from a trusted authority:

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/ssl_warning.jpg" width="400">

You can choose to override this, and it won't re-appear again until you change certificates or your device's hostname/IP changes.

### Start Web Agent

```bash
./run.sh \
  --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> \
  $(./autotag local_llm) \
    python3 -m local_llm.agents.web_chat --api=mlc \
      --model meta-llama/Llama-2-7b-chat-hf
```

You can then navigate your web browser to `https://HOSTNAME:8050` and unmute your microphone.

* The default port is 8050, but can be changed with `--web-port` (and `--ws-port` for the websocket port)
* To debug issues with client/server communication, use `--web-trace` to print incoming/outgoing websocket messages.
* During bot replies, the TTS model will pause output if you speak a few words in the mic to interrupt it.
* If you loaded a multimodal Llava model instead, you can drag-and-drop images from the client. 
  
## Live Llava

<a href="https://youtu.be/X-OXxPiUTuU" target="_blank"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava.gif"></a>

The [`VideoQuery`](agents/video_query.py) agent processes an incoming camera or video feed on prompts in a closed loop with Llava.  Navigate your browser to `https://<IP_ADDRESS>:8050` after launching it, proceed past the [SSL warning](#enabling-httpsssl), and see this [**demo walkthrough**](https://www.youtube.com/watch?v=dRmAGGuupuE) video on using the web UI. 

```bash
./run.sh $(./autotag local_llm) \
  python3 -m local_llm.agents.video_query --api=mlc \
    --model Efficient-Large-Model/VILA-2.7b \
    --max-context-len 768 \
    --max-new-tokens 32 \
    --video-input /dev/video0 \
    --video-output webrtc://@:8554/output
```

<a href="https://youtu.be/dRmAGGuupuE" target="_blank"><img width="750px" src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava_espresso.jpg"></a>

This uses [`jetson_utils`](https://github.com/dusty-nv/jetson-utils) for video I/O, and for options related to protocols and file formats, see [Camera Streaming and Multimedia](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md).  In the example above, it captures a V4L2 USB webcam connected to the Jetson (under the device `/dev/video0`) and outputs a WebRTC stream.

### Processing a Video File or Stream

The example above was running on a live camera, but you can also read and write a [video file or stream](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md) by substituting the path or URL to the `--video-input` and `--video-output` command-line arguments like this:

```bash
./run.sh \
  -v /path/to/your/videos:/mount
  $(./autotag local_llm) \
    python3 -m local_llm.agents.video_query --api=mlc \
      --model Efficient-Large-Model/VILA-2.7b \
      --max-context-len 768 \
      --max-new-tokens 32 \
      --video-input /mount/my_video.mp4 \
      --video-output /mount/output.mp4 \
      --prompt "What does the weather look like?"
```

This example processes and pre-recorded video (in MP4, MKV, AVI, FLV formats with H.264/H.265 encoding), but it also can input/output live network streams like [RTP](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#rtp), [RTSP](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#rtsp), and [WebRTC](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#webrtc) using Jetson's hardware-accelerated video codecs.

### NanoDB Integration

If you launch the [`VideoQuery`](agents/video_query.py) agent with the `--nanodb` flag along with a path to your NanoDB database, it will perform reverse-image search on the incoming feed against the database by re-using the CLIP embeddings generated by the VLM.

To enable this mode, first follow the [NanoDB tutorial](https://www.jetson-ai-lab.com/tutorial_nanodb.html) to download, index, and test the database.  Then launch VideoQuery like this:

```bash
./run.sh $(./autotag local_llm) \
  python3 -m local_llm.agents.video_query --api=mlc \
    --model Efficient-Large-Model/VILA-2.7b \
    --max-context-len 768 \
    --max-new-tokens 32 \
    --video-input /dev/video0 \
    --video-output webrtc://@:8554/output \
    --nanodb /data/nanodb/coco/2017
```

You can also tag incoming images and add them to the database using the panel in the web UI:

<a href="https://youtu.be/dRmAGGuupuE"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava_bear.jpg"></a>
> [Live Llava 2.0 - VILA + Multimodal NanoDB on Jetson Orin](https://youtu.be/X-OXxPiUTuU) (container: [`local_llm`](/packages/llm/local_llm#live-llava)) 

</details>