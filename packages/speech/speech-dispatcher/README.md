# speech-dispatcher

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

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
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`speech-dispatcher`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies |  |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag speech-dispatcher)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host speech-dispatcher:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag speech-dispatcher)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag speech-dispatcher) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build speech-dispatcher
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
