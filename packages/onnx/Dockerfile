#
# name: onnx
# category: ml
# config: config.py
# depends: [cmake, python, numpy, protobuf:apt]
# test: test.py
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ONNX_VERSION

RUN pip3 install --no-cache-dir --verbose git+https://github.com/onnx/onnx@${ONNX_VERSION}
#RUN pip3 install --no-cache-dir --verbose onnx
	
RUN pip3 show onnx && python3 -c 'import onnx; print(onnx.__version__)'
