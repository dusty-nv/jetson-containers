Running the `jupyerlab` container will automatically start a JupyterLab server in the background on port 8888, with default login password `nvidia`.  The JupyterLab server logs will be saved to `/data/logs/jupyter.log` if you need to inspect them (this location is automatically mounted under your `jetson-containers/data` directory)

To change the default settings, you can set the `$JUPYTER_ROOT`, `$JUPYTER_PORT`, `$JUPYTER_PASSWORD`, and `$JUPYTER_LOG` environment variables when starting the container like so:

```bash
jetson-containers run \
  --env JUPYTER_ROOT=/home/user \
  --env JUPYTER_PORT=8000 \
  --env JUPYTER_PASSWORD=password \
  --env JUPYTER_LOGS=/dev/null \
  $(autotag jupyterlab)
```

The [`/start_jupyter`](./start_jupyter) script is the default CMD that the container runs when it starts - however, if you don't want the JupyterLab server started by default, you can either add a different CMD in your own Dockerfile, or override it at startup:

```bash
# skip straight to the terminal instead of starting JupyterLab first
jetson-containers run /bin/bash
```

You can then still manually run the [`/start_jupyter`](./start_jupyter) script later when desired.
