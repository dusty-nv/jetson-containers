ARG BASE_IMAGE
FROM ${BASE_IMAGE}
    
# https://github.com/numpy/numpy/issues/18131#issuecomment-755438271
# ENV OPENBLAS_CORETYPE=ARMV8

RUN pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'
RUN pip3 install --upgrade --no-cache-dir --verbose numpy
RUN pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'
RUN pip3 show pandas && python3 -c 'import pandas; print(pandas.__version__)'

# Install additional libraries
RUN pip3 install --upgrade --no-cache-dir --verbose matplotlib ultralytics opencv-python super-globals

RUN pip3 show matplotlib && python3 -c 'import matplotlib; print(matplotlib.__version__)'
RUN pip3 show ultralytics && python3 -c 'import ultralytics; print(ultralytics.__version__)'
RUN pip3 show opencv-python && python3 -c 'import cv2; print(cv2.__version__)'
