
* these are the NVIDIA Riva [C++](https://github.com/nvidia-riva/cpp-clients) and [Python](https://github.com/nvidia-riva/python-clients) clients only (found under `/opt/riva/python-clients`)
* see [`riva_quickstart_arm64`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) from NGC to start the core Riva server container first
* Riva API reference docs:  https://docs.nvidia.com/deeplearning/riva/user-guide/docs/

### Start Riva Server

Before doing anything, you should download and run the Riva server container from [`riva_quickstart_arm64`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) using `riva_start.sh`

This will run locally on your Jetson Xavier or Orin device and is [supported on JetPack 5](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/support-matrix.html#embedded).  You can disable NLP/NMT in its `config.sh` and it will use around ~5GB of memory for ASR+TTS.  It's then recommended to test the system with [these examples](https://github.com/nvidia-riva/python-clients#asr) under `/opt/riva/python-clients`

You can also see this helpful video and guide from JetsonHacks for setting up Riva:  [**Speech AI on Jetson Tutorial**](https://jetsonhacks.com/2023/08/07/speech-ai-on-nvidia-jetson-tutorial/)

### List Audio Devices

This will print out a list of audio input/output devices that are connected to your system:

```bash
./run.sh --workdir /opt/riva/python-clients $(./autotag riva-client:python) \
   python3 scripts/list_audio_devices.py
```

You can refer to them in the steps below by either their device number or name.  Depending on the sample rate they support, you may also need to set `--sample-rate-hz` below to a valid frequency (e.g. `16000` `44100` `48000`)

### Streaming ASR

```bash
./run.sh --workdir /opt/riva/python-clients $(./autotag riva-client:python) \
   python3 scripts/asr/transcribe_mic.py --input-device=24 --sample-rate-hz=48000
```

You can find more ASR examples to run at https://github.com/nvidia-riva/python-clients#asr

### Streaming TTS

```bash
./run.sh --workdir /opt/riva/python-clients $(./autotag riva-client:python) \
   python3 scripts/tts/talk.py --stream --output-device=24 --sample-rate-hz=48000 \
     --text "Hello, how are you today? My name is Riva." 
```

You can set the `--voice` argument to one of the [available voices](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html#voices) (the default is `English-US.Female-1`)

Also, you can customize the rate, pitch, and pronunciation of individual words/phrases by including [inline SSML](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-basics-customize-ssml.html#customizing-riva-tts-audio-output-with-ssml) in your text.

### Loopback

To feed the live ASR transcript into the TTS and have it speak your words back to you:

```bash
./run.sh --workdir /opt/riva/python-clients $(./autotag riva-client:python) \
   python3 scripts/loopback.py --input-device=24 --output-device=24 --sample-rate-hz=48000
```