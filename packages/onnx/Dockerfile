#
# name: onnx
# category: ml
# config: config.py
# depends: [cmake, python, numpy]
# test: test.py
# notes: protobuf_apt is added as a dependency on JetPack 4 (newer versions of onnx build it in-tree)
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ONNX_VERSION

RUN pip3 install --no-cache-dir --verbose git+https://github.com/onnx/onnx@${ONNX_VERSION}
#RUN pip3 install --no-cache-dir --verbose onnx
	
RUN pip3 show onnx && python3 -c 'import onnx; print(onnx.__version__)'
