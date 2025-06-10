
* minigpt4.cpp from https://github.com/Maknee/minigpt4.cpp with CUDA enabled (found under `/opt/minigpt4.cpp`)
* original MiniGPT-4 models and project are from https://github.com/Vision-CAIR/MiniGPT-4

To start the MiniGPT4 container and webserver with the [recommended models](https://github.com/Maknee/minigpt4.cpp/tree/master#3-obtaining-the-model), run this command:

```bash
./run.sh $(./autotag minigpt4) /bin/bash -c 'cd /opt/minigpt4.cpp/minigpt4 && python3 webui.py \
  $(huggingface-downloader --type=dataset maknee/minigpt4-13b-ggml/minigpt4-13B-f16.bin) \
  $(huggingface-downloader --type=dataset maknee/ggml-vicuna-v0-quantized/ggml-vicuna-13B-v0-q5_k.bin)'
```

Then navigate your browser to `http://HOSTNAME:7860`

### Inference Benchmark

```
./run.sh --workdir=/opt/minigpt4.cpp/minigpt4/ $(./autotag minigpt4) /bin/bash -c \
  'python3 benchmark.py \
    $(huggingface-downloader --type=dataset maknee/minigpt4-13b-ggml/minigpt4-13B-f16.bin) \
    $(huggingface-downloader --type=dataset maknee/ggml-vicuna-v0-quantized/ggml-vicuna-13B-v0-q5_k.bin) \
    --prompt "What does the sign say?" --prompt "How far is the exit?" --prompt "What would happen next?" \
    --image /data/images/hoover.jpg \
    --run 3 \
    --save /data/minigpt4.csv'
```


