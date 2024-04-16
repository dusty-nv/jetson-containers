
* llama-index from https://www.llamaindex.ai/

### Container with samples

```bash
jetson-containers run $(./autotag llama-index:samples)
```

#### Data set up for the sample

On the Docker host console, copy the L4T-README text files to jetson-container's `/data` directory.

```bash
cd jetson-containers
mkdir -p data/documents/L4T-README
cp /media/jetson/L4T-README/*.txt data/documents/L4T-README/
```

#### How to run the sample

Once in the container terminal, first run the Ollama server.

```bash
/bin/ollama serve > /var/log/ollama.log &
```

Hit enter to come back to console, then download the Llama2 model using `ollama`.

```bash
/bin/ollama pull llama2
```

Then, run the sample script to ask Jetson related questions (***"With USB device mode, what IP address Jetson gets? Which file should be edited in order to change the IP address assigned to Jetson?"***)to let the Llama-2 model answer based on the provided README files.

```bash
python3 llamaindex_starter.py
```

It should answer something like this;

```text
Based on the context provided, the static IP address assigned to Jetson is 192.168.55.100. To change the IP address assigned to Jetson, you should edit the "Property" section of the "Remote NDIS Compatible Device" interface in the Network Connections settings on your host system. The file that should be edited is /opt/nvidia/l4t-usb-device-mode/nv-l4t-usb-device-mode-config.sh on Jetson.
```



