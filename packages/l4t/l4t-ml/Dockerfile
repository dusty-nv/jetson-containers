#---
# name: l4t-ml
# group: ml
# depends: [pytorch, torchvision, torchaudio, tensorflow2, opencv, pycuda, cupy, onnxruntime, numba, gstreamer, jupyterlab]
# test: [test_pandas.py, test_scipy.py, test_sklearn.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install pandas
