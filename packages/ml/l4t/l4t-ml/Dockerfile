#---
# name: l4t-ml
# group: ml
# depends: [
#   tensorflow2, jax,
#   pytorch, torchvision, torchaudio, torchao,
#   pycuda, cupy, cuda-python,
#   onnxruntime, numba,
#   opencv, gstreamer, jupyterlab]
# test: [test_pandas.py, test_scipy.py, test_sklearn.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN uv pip install scipy
RUN uv pip install scikit-learn
RUN uv pip install pandas
