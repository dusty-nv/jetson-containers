# CUDA Stack - Consolidated Library Installation

## What is cudastack?

`cudastack` is a **consolidation package** that installs multiple CUDA libraries (cuDNN, NCCL, TensorRT, CUTLASS, etc.) in **ONE RUN command** to avoid Docker's "max depth exceeded" error.

## Important: cudastack vs cuda

```
cuda/                           cudastack/
├── Dockerfile                  ├── Dockerfile (ONE RUN for all libs)
├── Dockerfile.pip              ├── config.py (variants)
├── Dockerfile.builtin          └── install/ & build/ scripts
├── Dockerfile.samples
└── config.py
    ├── cuda_package()          cudastack DEPENDS ON cuda
    ├── cuda_builtin()          (doesn't replace it!)
    ├── cuda_samples()
    └── pip_cache()
```

**Key Points:**

1. ✅ **`cuda` package** - Installs CUDA toolkit, provides pip_cache, samples, builtin variants
2. ✅ **`cudastack` package** - Builds ON TOP of `cuda`, adds cuDNN/NCCL/TensorRT/etc. in ONE layer
3. ✅ **pip_cache, cuda_samples, cuda_builtin** - Still work! They're part of `cuda`, not `cudastack`

## The Problem It Solves

### Before (Separate Packages):
```
Layer 1: cuda
Layer 2: cudnn
Layer 3: tensorrt
Layer 4: nccl
Layer 5: cutlass
...
Layer 50+: "max depth exceeded" ❌
```

### After (cudastack):
```
Layer 1: cuda (with pip_cache, samples, etc.)
Layer 2: cudastack (cudnn + nccl + tensorrt + cutlass in ONE RUN)
Layer 3: your-app
Result: Works! ✅
```

## Architecture

```
BASE_IMAGE
    ↓
cuda:12.4 (from cuda/Dockerfile)
    ├── pip_cache:cu124 (from cuda/Dockerfile.pip)
    └── cuda:12.4-samples (from cuda/Dockerfile.samples)
    ↓
cudastack:minimal (from cudastack/Dockerfile)
    ├── cuDNN
    ├── NCCL
    └── (all installed in ONE RUN)
    ↓
your-app
```

## Variants

| Variant | Includes | Use Case |
|---------|----------|----------|
| `cudastack:minimal` | cuDNN + NCCL | Training, inference |
| `cudastack:standard` | + TensorRT | Optimized inference |
| `cudastack:full` | + CUTLASS + GDRCopy | Research, development |

## Usage

### Build with jetson-containers:

```bash
# Build minimal stack
./build.sh cudastack:minimal

# Build standard (with TensorRT)
./build.sh cudastack:standard

# Build full (with CUTLASS)
./build.sh cudastack:full

# Use as base for your app
./build.sh --base cudastack:minimal pytorch
```

### The cudastack Dockerfile does ONE RUN:

```dockerfile
# Copy all install/build scripts
COPY install/*.sh /tmp/cudastack/install/
COPY build/*.sh /tmp/cudastack/build/

# ONE RUN command calls all scripts conditionally
RUN set -ex && \
    if [ "$WITH_CUDNN" = "1" ]; then \
        ./install/install_cudnn.sh; \
    fi && \
    if [ "$WITH_NCCL" = "1" ]; then \
        ./install/install_nccl.sh; \
    fi && \
    if [ "$WITH_TENSORRT" = "1" ]; then \
        ./install/install_tensorrt.sh; \
    fi && \
    # ... more components ...
    rm -rf /tmp/cudastack
```

**Key**: All library installations happen in this ONE RUN = ONE Docker layer!

## Setup

The install scripts need to be populated from existing packages:

```bash
cd packages/cuda/cudastack
./setup-scripts.sh
```

Then review and complete the scripts in `install/` and `build/`.

## What About pip_cache, samples, builtin?

These are **still available** from the `cuda` package:

```bash
# pip_cache is automatically included by cuda package
./build.sh cuda:12.4
# This creates both:
# - cuda:12.4 (main CUDA toolkit)
# - pip_cache:cu124 (pip caching)

# CUDA samples work as before
./build.sh cuda:12.4-samples

# Builtin works as before (for JetPack 4-5)
./build.sh cuda:11.4  # Uses Dockerfile.builtin when CUDA pre-installed
```

The `cuda` package's `config.py` returns multiple packages:
```python
# In cuda/config.py
def cuda_package(version, url, ...):
    cuda = package.copy()
    cuda['name'] = f'cuda:{version}'
    
    cuda_pip = pip_cache(version, requires)  # Creates pip_cache package
    
    return cuda, cuda_pip  # Returns TWO packages!
```

**cudastack doesn't need to duplicate this** - it just depends on `cuda` being installed first.

## Component Toggles

Controlled by `WITH_*` build args (set in config.py):

- ✅ `WITH_CUDNN=1` - cuDNN (always on)
- ✅ `WITH_NCCL=1` - NCCL (always on)
- 🎛️ `WITH_TENSORRT` - TensorRT (on in standard/full)
- 🎛️ `WITH_CUTLASS` - CUTLASS (on in full only)
- 🎛️ `WITH_GDRCOPY` - GDRCopy (on for Tegra)
- ⚙️ `WITH_CUDSS=0` - cuDSS (off by default)
- ⚙️ `WITH_CUSPARSELT=0` - cuSPARSELt (off by default)
- ⚙️ `WITH_CUTENSOR=0` - cuTENSOR (off by default)
- ⚙️ `WITH_NVPL=0` - NVPL (off by default)
- ⚙️ `WITH_NVSHMEM=0` - NVSHMEM (off by default)

## How config.py Works

Following the jetson-containers pattern:

```python
def cuda_stack(name, with_tensorrt=False, with_cutlass=False, requires=None):
    """Generate a cudastack package variant"""
    pkg = package.copy()
    pkg['name'] = name
    pkg['depends'] = ['cuda']  # Depends ON cuda, doesn't replace it
    
    pkg['build_args'] = {
        'WITH_CUDNN': '1',
        'WITH_TENSORRT': '1' if with_tensorrt else '0',
        'WITH_NCCL': '1',
        # ... etc
    }
    
    return pkg

# Define variants
package = [
    cuda_stack('cudastack:minimal', with_tensorrt=False),
    cuda_stack('cudastack:standard', with_tensorrt=True),
    cuda_stack('cudastack:full', with_tensorrt=True, with_cutlass=True),
]
```

## Comparison with Individual Packages

| Approach | Layers | Build Time | Complexity |
|----------|--------|------------|------------|
| **Individual packages** | 50+ | Longer | Simple per-package |
| **cudastack (consolidated)** | 2-3 | Faster | One complex layer |

Both approaches work! Use `cudastack` when you need multiple libraries and want to avoid layer limits.

## See Also

- [cuda package](../cuda/) - CUDA toolkit, pip_cache, samples, builtin
- [cudnn package](../cudnn/) - Individual cuDNN package
- [tensorrt package](../tensorrt/) - Individual TensorRT package (has Dockerfile.deb and Dockerfile.tar)
- [nccl package](../nccl/) - Individual NCCL package

## Summary

✅ **cudastack is a consolidation layer** on top of cuda  
✅ **pip_cache, samples, builtin still work** from the cuda package  
✅ **ONE RUN command** installs all libraries to avoid layer limits  
✅ **Follows jetson-containers patterns** with config.py and variants  
