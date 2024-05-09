
* whisper_streaming from https://github.com/ufal/whisper_streaming

### Testing real-time simulation from audio file

Once in container;

```bash
cd whisper_streaming/
python3 whisper_online.py --model tiny.en --lan en --backend faster-whisper /data/audio/asr/Micro-Machine.wav
```

If you want to save all the output to file.

```bash
time python3 whisper_online.py --model large-v3 --lan en --backend faster-whisper /data/audio/asr/Micro-Machine.wav 2>&1 | tee -a /data/audio/asr/MM_large-v3_En.logws
```

### Testing server mode -- real-time from mic

#### Terminal 1: Inside the container

```bash
cd whisper_streaming/
python3 whisper_online_server.py --port 43001 --model medium.en
```

#### Terminal 2: Outside the container

On another terminal, just on the host (not in container), first check if your system can find a microphone.

```bash
arcord -l
```

The output may contain list like this, and it confirms it is seen as `hw:2,0`

```
card 2: Headset [Logitech USB Headset], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

You can execute the following to netcat the captured audio to `localhost:43001` so that the server running in the container can process.

```bash
arecord -f S16_LE -c1 -r 16000 -t raw -D hw:2,0 | nc localhost 43001
```

### Benchmark




