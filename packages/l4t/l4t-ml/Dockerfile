#
# name: l4t-ml
# category: ml
# depends: [pytorch, torchvision, torchaudio, tensorflow2, opencv, pycuda, cupy, onnxruntime, jupyterlab]
# test: [test_numba.py, test_pandas.py, test_scipy.py, test_sklearn.py]
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip3 install --no-cache-dir --verbose scipy
RUN pip3 install --no-cache-dir --verbose scikit-learn
RUN pip3 install --no-cache-dir --verbose pandas
RUN pip3 install --no-cache-dir --verbose numba
