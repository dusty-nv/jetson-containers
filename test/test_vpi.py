
print('testing VPI (Python)...')
import vpi

# test sample from https://docs.nvidia.com/vpi/tutorial_python.html
# and /opt/nvidia/vpi2/samples/01-convolve_2d
import numpy as np
from PIL import Image

input = vpi.asimage(np.asarray(Image.open('/test/data/test_0.jpg')))
input = input.convert(vpi.Format.U8, backend=vpi.Backend.CUDA)

backends = [
    ('cpu', vpi.Backend.CPU),
    ('cuda', vpi.Backend.CUDA),
    ('pva', vpi.Backend.PVA)
]

for backend_name, backend in backends:
    print(f"testing backend '{backend_name}'")
    output = input.box_filter(5, border=vpi.Border.ZERO, backend=backend)
    Image.fromarray(output.cpu()).save(f'/test/data/test_vpi_{backend_name}.jpg')
     
print('VPI (Python) OK\n')
