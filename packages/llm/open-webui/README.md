### Open WebUI with CUDA Support for Jetson ( requires Python 3.11+ )

- Example build command for CUDA 12.9 on Jetson with Ubuntu 24.04, Python 3.12, and PyTorch 2.9.1:
  - Pass `USE_OLLAMA=on` as the example above to build Ollama in the same container.

```bash
LSB_RELEASE=24.04 CUDA_VERSION=12.9 PYTHON_VERSION=3.12 PYTORCH_VERSION=2.9.1 USE_OLLAMA=on jetson-containers build open-webui
  ```

- Example run command:
  - Map a persistent directory on the host to `/app/backend/data` in the container to store database and settings.
  - Other env params can be passed if needed, like `OLLAMA_KEEP_ALIVE=15m` or `OLLAMA_CONTEXT_LENGTH=32000`: 

```bash
jetson-containers run -v /path/to/your/open-webui/persistent/dir:/app/backend/data open-webui
```

- After Open WebUI starts, access it via `http://<jetson-ip>:8080` in your web browser.
