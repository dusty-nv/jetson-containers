
* minigpt4.cpp from https://github.com/Maknee/minigpt4.cpp with CUDA enabled (found under `/opt/minigpt4.cpp`)

To start the web server with the [recommended models](https://github.com/Maknee/minigpt4.cpp/tree/master#3-obtaining-the-model), run this:

```bash
./run.sh --workdir=/opt/minigpt4.cpp/minigpt4 $(./autotag minigpt4) /bin/bash -c 'python3 webui.py \
  $(huggingface-downloader --type=dataset maknee/minigpt4-13b-ggml/minigpt4-13B-f16.bin) \
  $(huggingface-downloader --type=dataset maknee/ggml-vicuna-v0-quantized/ggml-vicuna-13B-v0-q5_k.bin)'
```

Then navigate your browser to `http://HOSTNAME:7860`
