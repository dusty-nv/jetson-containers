
print('testing protobuf...')

import google.protobuf
print('protobuf version: ' + str(google.protobuf.__version__))

# verify implementation is cpp
from google.protobuf.internal import api_implementation

protobuf_implementation = str(api_implementation.Type())

print(f'protobuf default API implementation: {str(api_implementation._default_implementation_type)}')
print(f'protobuf active API implementation:  {protobuf_implementation}')

if protobuf_implementation != "cpp":
    raise ValueError(f'expected protobuf to have cpp implementation, but instead it has {protobuf_implementation} implementation')
    
print('protobuf OK\n')
