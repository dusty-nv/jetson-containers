
from jetson_containers import CUDA_VERSION, package_depends
from packaging.version import Version

# by default NVILA requires deepspeed==0.9.5 for the .train packages,
# but this conflicts with numpy2 - so only enforce it where numpy<2
if CUDA_VERSION <= Version('12.6'):
    package_depends(package, 'deepspeed:0.9.5')

