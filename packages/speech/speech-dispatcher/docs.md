# speech-dispatcher in container and PulseAudio Docker setting.

This document outlines the steps to modify the PulseAudio configuration on Docker host (Jetson) so to run an audio enabled application in continer.
It can be referenced when one tries to enable similar PulseAudio based application inside a container.

## Step 1. PulseAudio Configuration on Host

- Share PulseAudio socket file (`/run/user/1000/pulse/native`)
- Set `PULSE_SERVER` env variable in the container
- Allow root access to the socket file

To output sound from your Docker container while using the host's sound device, you can map the appropriate sound devices and audio services from the host to the container. 

We use PulseAudio from within the container. To do this, we need share the PulseAudio socket between the host and the container.

### jetson-containers `run.sh` modification

Expand the `docker run` options for sound like this.

```bash
   --device /dev/snd \
   -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
   -v ${XDG_RUNTIME_DIR}/pulse:${XDG_RUNTIME_DIR}/pulse \
```

Option: If not working, further expand like following and try.

```bash
   --device /dev/snd \
   -e XDG_RUNTIME_DIR=/run/user/1000 \
   -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
   -v ${XDG_RUNTIME_DIR}/pulse:${XDG_RUNTIME_DIR}/pulse \
   -v /etc/machine-id:/etc/machine-id \
   --group-add audio \
```

### Edit ``/etc/pulse/default.pa``

We operate as `root` in container, so we need to mofidy PusleAudio configuraiton file so that it allows the root user access to the socket file.

```bash
sudo vi /etc/pulse/default.pa
```

Find the section loading `module-native-protomocl-unix` and add `auth-anonymous=1` 

```bash
### Load several protocols
.ifexists module-esound-protocol-unix.so
load-module module-esound-protocol-unix auth-anonymous=1
.endif
load-module module-native-protocol-unix auth-anonymous=1
```

Restart Pulse Audio service.

```bash
pulseaudio --kill
pulseaudio --start
```

## Step 2. Container setup

Inside the container, install `speech-dispatcher` and run its server inside the container (as opposed to running the `speech-dispatcher` server on the host).

Here is the Dockefile snippet.

```dockerfile
RUN apt-get update && \
    apt install -y speech-dispatcher alsa-utils && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

CMD speech-dispatcher --spawn  && /bin/bash
```

Notice it's starting the `speech-dispatcher` server as a background process within the container with `speech-dispatcher --spawn`.

## Step 3. Check the default audio sink device

Reboot the host system to make sure PulseAudio starts with the modified config file.

Then, on the host, check the default audio sink.

```bash
pactl info

```

If need to be set, `set-default-sink`.

```bash
pactl list short sinks
pactl set-default-sink [SINK_NAME_OR_INDEX]
```

## Step 4. Run the container

```bash
jetson-containers run $(autotag speech-dispatcher)
```

Once inside the container;

```bash
spd-say "Hello world"
```

You should here TTS audio coming out from your host audio device.

## Troubleshooting

### Restart PulseAudio with debug output

```
pulseaudio -k
pulseaudio -vvv
```

### Alsa layer test

ALSA should work in your container just with `/dev/snd` mapped.

Test with `speaker-test -t wav -c 2` (alsa-utils).