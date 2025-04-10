
from jetson_containers import L4T_VERSION

if L4T_VERSION.major >= 34:  # JetPack 5
    # https://github.com/onnx/onnx/issues/5346
    package['build_args'] = {'ONNX_VERSION': 'main'}
else:
    # onnx 1.15:  SyntaxError: future feature annotations is not defined (/onnx/__init__.py", line 5)
	# onnx 1.14:  onnx/defs/shape_inference.h:828:8: error: 'transform' is not a member of 'std'
	# onnx 1.13:  Could not find a version that satisfies the requirement protobuf<4,>=3.20.2 (from onnx)
	# onnx 1.12:  /onnx/defs/sequence/defs.cc:675:40: error: no match for 'operator[]' (operand types are 'google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>' and 'int')
	# onnx 1.11:  installs onnx-1.11.0-cp36-cp36m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl without building it
    package['build_args'] = {'ONNX_VERSION': 'v1.11.0'}
    package['depends'].append('protobuf:apt')