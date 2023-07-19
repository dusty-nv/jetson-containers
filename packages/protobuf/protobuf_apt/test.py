#!/usr/bin/env python3
print('testing protobuf (apt)...')

import google.protobuf
print('protobuf version: ' + str(google.protobuf.__version__))

# verify implementation type
from google.protobuf.internal import api_implementation
from packaging.version import Version

protobuf_implementation = str(api_implementation.Type())

if Version(google.protobuf.__version__).major < 4:
    print(f'protobuf default API implementation: {str(api_implementation._default_implementation_type)}')
    
print(f'protobuf active API implementation:  {protobuf_implementation}')

print('protobuf (apt) OK\n')
