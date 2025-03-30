from jetson_containers import CUDA_VERSION
from packaging.version import Version

def sudonim(tag=None, patches=None, requires=None, depends=None):
    pkg = package.copy()

    if tag:
        pkg['name'] = f'sudonim:{tag}'

    if requires:
        pkg['requires'] = requires   

    if depends:
        pkg['depends'] = pkg['depends'] + depends

    if patches:
        pkg['build_args'] = {
            'SUDONIM_PATCH_DIR': patches,
        }

    return pkg

package = [
    package.copy(),

    # for built-in endpoint using HF Transformers and torchao / bitsandbytes quantization
    sudonim(tag='hf', patches='250314', depends=['transformers', 'torchao', 'bitsandbytes', 'flash-attention', 'opencv'])
]

