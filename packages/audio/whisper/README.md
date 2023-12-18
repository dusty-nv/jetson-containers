# whisper

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


## Jupyter notebooks

Inside the container, you find the Whisper original notebooks (`LibriSpeech.ipynb`, `Multilingual_ASR.ipynb`) and the extra notebook (`record-and-transcribe.ipynb`) added by this `jetson-containers` package under the following directory.

`/opt/whisper/notebooks`

## Jupyter Lab setup

This container has a default run command that will automatically start the Jupyter by `CMD` command in Dockerfile like this:

```bash
CMD /bin/bash -c "jupyter lab --ip 0.0.0.0 --port 8888  --certfile=mycert.pem --keyfile mykey.key --allow-root &> /var/log/jupyter.log" & \
	echo "allow 10 sec for JupyterLab to start @ https://$(hostname -I | cut -d' ' -f1):8888 (password nvidia)" && \
	echo "JupterLab logging location:  /var/log/jupyter.log  (inside the container)" && \
	/bin/bash
```

Open your web browser and access `http://HOSTNAME:8888`.

It is enabling HTTPS (SSL) connection, so you will see a warning message like this.

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/chrome_ssl_cert.png" width="600px">

Press "**Advanced**" button and then press "**Proceed to <IP_ADDRESS> (unsafe)**" to proceed.

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/chrome_ssl_advanced.png" width="600px">

HTTPS (SSL) connection is needed to allow `ipywebrtc` widget to have access to the microphone (for `record-and-transcribe.ipynb`).

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`whisper`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`whisper_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/whisper_jp60.yml?label=whisper:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/whisper_jp60.yml) [![`whisper_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/whisper_jp51.yml?label=whisper:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/whisper_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`cuda`](/packages/cuda/cuda) [`python`](/packages/python) [`numpy`](/packages/numpy) [`numba`](/packages/numba) [`cudnn`](/packages/cuda/cudnn) [`tensorrt`](/packages/tensorrt) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchaudio`](/packages/pytorch/torchaudio) [`rust`](/packages/rust) [`jupyterlab`](/packages/jupyterlab) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/whisper:r35.3.1`](https://hub.docker.com/r/dustynv/whisper/tags) `(2023-12-12, 6.1GB)`<br>[`dustynv/whisper:r35.4.1`](https://hub.docker.com/r/dustynv/whisper/tags) `(2023-12-11, 6.1GB)`<br>[`dustynv/whisper:r36.2.0`](https://hub.docker.com/r/dustynv/whisper/tags) `(2023-12-12, 7.9GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/whisper:r35.3.1`](https://hub.docker.com/r/dustynv/whisper/tags) | `2023-12-12` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/whisper:r35.4.1`](https://hub.docker.com/r/dustynv/whisper/tags) | `2023-12-11` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/whisper:r36.2.0`](https://hub.docker.com/r/dustynv/whisper/tags) | `2023-12-12` | `arm64` | `7.9GB` |

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
./run.sh $(./autotag whisper)

# or explicitly specify one of the container images above
./run.sh dustynv/whisper:r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/whisper:r36.2.0
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag whisper)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag whisper) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh whisper
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
