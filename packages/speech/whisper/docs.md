
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
