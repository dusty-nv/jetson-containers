#!/bin/bash
python3 - <<EOF
print('testing MInference...')
from minference import MInferenceConfig
supported_attn_types = MInferenceConfig.get_available_attn_types()
supported_kv_types = MInferenceConfig.get_available_kv_types()
print('Supported attention types:', supported_attn_types)
print('Supported kv types:', supported_kv_types)

from minference import get_support_models
get_support_models()
print('MInference OK\\n')
EOF
