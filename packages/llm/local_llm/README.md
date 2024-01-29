# local_llm

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


<a href="https://www.youtube.com/watch?v=9ObzbbBTbcc"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_llava_clip.gif"></a>

* Optimized LLM inference engine with support for AWQ and MLC quantization, multimodal agents, and live ASR/TTS.
* Web UI server using Flask, WebSockets, WebAudio, HTML5, Bootstrap5.

> Modes to Run:
>  * [Text Chat](#text-chat)
>  * [Multimodal Chat](#multimodal-chat)
>  * [Voice Chat](#voice-chat)
>  * [Live Llava](#live-llava)

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
>   * to `"model_type": "llama",` <br/>
>
> Then re-run the command above - the quantization tools will then treat it like a Llama model (which it is)

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
  
## Live Llava

<a href="https://youtu.be/X-OXxPiUTuU" target="_blank"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava.gif"></a>

The [`VideoQuery`](agents/video_query.py) agent processes an incoming camera or video feed on prompts in a closed loop with Llava.  

```bash
./run.sh \
  -e HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> \
  -e SSL_KEY=/data/key.pem \
  -e SSL_CERT=/data/cert.pem \
  $(./autotag local_llm) \
	python3 -m local_llm.agents.video_query --api=mlc --verbose \
	  --model liuhaotian/llava-v1.5-7b \
	  --max-new-tokens 32 \
	  --video-input /dev/video0 \
	  --video-output webrtc://@:8554/output \
	  --prompt "How many fingers am I holding up?"
```
> see the [Enabling HTTPS/SSL](#enabling-httpsssl) section above to generate self-signed SSL certificates for enabling client-side browser webcams.

This uses [`jetson_utils`](/packages/jetson_utils) for video I/O, and for options related to camera protocols and streaming, see [Camera Streaming and Multimedia](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md).  In the example above, it captures a V4L2 USB webcam connected to the Jetson (under the device `/dev/video0`) and outputs a WebRTC stream that can be viewed at `https://HOSTNAME:8554`.  When HTTPS/SSL is enabled, it can also capture from the browser's webcam over WebRTC.

The `--prompt` can be specified multiple times, and changed at runtime by pressing the number of the prompt followed by enter on the terminal's keyboard (for example, <kbd>1</kbd> + <kbd>Enter</kbd> for the first prompt).  These are the default prompts when no `--prompt` is specified:

1. Describe the image concisely.
2. How many fingers is the person holding up?
3. What does the text in the image say?
4. There is a question asked in the image.  What is the answer?

Future versions of this demo will have the prompts dynamically editable in the web UI.

## Tested Models

Llama 2:

* [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
* [`meta-llama/Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
* [`meta-llama/Llama-2-70b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)

LLaVA:  

* [`liuhaotian/llava-v1.5-7b`](https://huggingface.co/liuhaotian/llava-v1.5-7b)
* [`liuhaotian/llava-v1.5-13b`](https://huggingface.co/liuhaotian/llava-v1.5-13b)

Any fine-tuned version of Llama or Llava that shares the same architecture (or that is supported by the quantization API you have selected) should be compatible, like Vicuna, CodeLlama, ect.  See [here](https://github.com/mlc-ai/mlc-llm/tree/main/mlc_llm/relax_model) for the MLC model architectures.

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`local_llm`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`local_llm_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/local_llm_jp60.yml?label=local_llm:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/local_llm_jp60.yml) [![`local_llm_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/local_llm_jp51.yml?label=local_llm:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/local_llm_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/rust) [`transformers`](/packages/llm/transformers) [`mlc`](/packages/llm/mlc) [`riva-client:python`](/packages/audio/riva-client) [`opencv`](/packages/opencv) [`gstreamer`](/packages/gstreamer) [`jetson-inference`](/packages/jetson-inference) [`torch2trt`](/packages/pytorch/torch2trt) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/local_llm:r35.2.1`](https://hub.docker.com/r/dustynv/local_llm/tags) `(2023-12-22, 9.6GB)`<br>[`dustynv/local_llm:r35.3.1`](https://hub.docker.com/r/dustynv/local_llm/tags) `(2024-01-27, 10.1GB)`<br>[`dustynv/local_llm:r35.4.1`](https://hub.docker.com/r/dustynv/local_llm/tags) `(2023-12-22, 9.6GB)`<br>[`dustynv/local_llm:r36.2.0`](https://hub.docker.com/r/dustynv/local_llm/tags) `(2024-01-27, 11.3GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/local_llm:r35.2.1`](https://hub.docker.com/r/dustynv/local_llm/tags) | `2023-12-22` | `arm64` | `9.6GB` |
| &nbsp;&nbsp;[`dustynv/local_llm:r35.3.1`](https://hub.docker.com/r/dustynv/local_llm/tags) | `2024-01-27` | `arm64` | `10.1GB` |
| &nbsp;&nbsp;[`dustynv/local_llm:r35.4.1`](https://hub.docker.com/r/dustynv/local_llm/tags) | `2023-12-22` | `arm64` | `9.6GB` |
| &nbsp;&nbsp;[`dustynv/local_llm:r36.2.0`](https://hub.docker.com/r/dustynv/local_llm/tags) | `2024-01-27` | `arm64` | `11.3GB` |

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
./run.sh $(./autotag local_llm)

# or explicitly specify one of the container images above
./run.sh dustynv/local_llm:r35.3.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/local_llm:r35.3.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag local_llm)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag local_llm) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh local_llm
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
