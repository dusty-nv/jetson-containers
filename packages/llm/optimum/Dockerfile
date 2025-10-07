#---
# name: optimum
# group: llm
# config: config.py
# depends: [onnxruntime, transformers]
# test: test.py
# notes: https://github.com/huggingface/optimum
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN uv pip install optimum[exporters,onnxruntime]

RUN uv pip uninstall onnxruntime onnxruntime-gpu && \
    uv pip install /opt/onnxruntime_gpu-*.whl

# ImportError: `onnxruntime-gpu` is installed, but GPU dependencies are not loaded.
RUN PYTHON_DIST_PACKAGES=$(python3 -c 'import sys; print(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")') && \
    sed -i 's/if "ORT_CUDA" not in file_string or "ORT_TENSORRT" not in file_string:/if False:/g' $PYTHON_DIST_PACKAGES/optimum/onnxruntime/utils.py
