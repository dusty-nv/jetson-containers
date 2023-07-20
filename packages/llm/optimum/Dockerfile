#---
# name: optimum
# category: llm
# depends: [onnxruntime, transformers]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip3 install --no-cache-dir --verbose optimum[exporters,onnxruntime]

RUN pip3 uninstall -y onnxruntime onnxruntime-gpu && \
    pip3 install --no-cache-dir --verbose /opt/onnxruntime_gpu-*.whl
    
# ImportError: `onnxruntime-gpu` is installed, but GPU dependencies are not loaded.
RUN PYTHON_DIST_PACKAGES=$(python3 -c 'import sys; print(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")') && \
    sed -i 's/if "ORT_CUDA" not in file_string or "ORT_TENSORRT" not in file_string:/if False:/g' $PYTHON_DIST_PACKAGES/optimum/onnxruntime/utils.py
