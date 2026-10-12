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

## Overriding the TensorRT version (JetPack 6)

On L4T r36, set `TENSORRT_VERSION` to pick a different TensorRT than the CUDA default
(`10.7`, `10.3`, `10.0`; `10.7.0` is read as `10.7`). Only the TensorRT version/URL change; cuDNN
stays on the CUDA-based default (e.g. 9.3.0 for CUDA 12.6).

- The TensorRT tarball has to match the CUDA version: `10.7` and `10.3` (built for CUDA 12.6) are
  accepted with CUDA 12.6 and 12.8, `10.0` (CUDA 12.4) with CUDA 12.4 to 12.8. Any other
  combination fails with the versions supported for that CUDA.
- On other targets (r39+, SBSA, x86) the variable is ignored, with a warning.
- The image tag does not include the TensorRT version, so `cudastack:standard` built with and
  without the override get the same tag. Don't push an overridden build under the default tag.

```bash
TENSORRT_VERSION=10.7 jetson-containers build cudastack:standard
```

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

## Chain-aware selection (`cuda_libs`)

By default a `cudastack` stage installs its whole preset (for `standard`: TensorRT, cuDSS,
cuSPARSELt, and so on). Packages can instead declare the CUDA libraries they need, and the
builder installs only what the chain being built needs. It still does this in the same single
`RUN`.

**Declare** in the package's Dockerfile header:

```dockerfile
#---
# name: jax
# depends: [cuda, cudastack:standard, numpy, onnx, llvm:22]
# cuda_libs: [cudnn, nccl]
#---
```

Valid names: `cudnn`, `tensorrt`, `nccl`, `cudss`, `cusparselt`, `cutensor`, `nvshmem`,
`nvpl`, `gdrcopy`. An unknown name makes that package fail to load: an error naming the
package and the bad value is printed during the scan, and the package is then unavailable
to build. List what the package needs **at runtime and at build time**, including for its
`-builder` variant -- e.g. for pytorch that means every `USE_*=1` library in `build.sh`,
plus its BLAS (NVPL on SBSA).

**How the set is computed** (`jetson_containers/cudastack_chain.py`, called from
`build_container()` after the chain is resolved):

- Only packages whose own dependencies include `cudastack` count. `numpy`, `cmake` and
  other packages that don't use it are ignored.
- The set is the union of those packages' `cuda_libs`. cuDNN is always included.
- A cudastack dependent that declares **nothing** needs the full preset. So until every
  cudastack package in a chain declares, that chain builds exactly what it builds today.
  For example `opencv pytorch torchvision` still gets the full preset, because `opencv`
  and `torchvision` don't declare `cuda_libs` yet, while `pytorch` alone narrows.
- **Narrow-only:** the result never turns on a library that the variant's preset has off.
- Building `cudastack` by itself (nothing after it in the chain) uses the full preset.
- `WITH_CUTLASS` is not part of the presets, so it is never narrowed.

**Stage tag.** A narrowed stage is tagged with its sorted set, for example
`...-cudastack_cudnn-nccl` instead of `...-cudastack_standard`. That way different sets
don't overwrite each other -- the stage tag differs per build target name. Docker's own
layer cache reuses identical stages (same base image + same build args) regardless of the
tag, so chains that compute the same set still share cached layers. When the
set equals the preset, the tag is unchanged (`-cudastack_standard`). The builder logs one
line per build, for example:
`cudastack chain-aware: cudastack:standard -> [cudnn, nccl] (stage ...-cudastack_cudnn-nccl; CUDASTACK_CHAIN_AWARE=0 to disable)`.

**Opt-out.** Set `CUDASTACK_CHAIN_AWARE=0` (environment variable, or
`--build-args CUDASTACK_CHAIN_AWARE:0`) to always install the full preset with today's tags.
**Any pipeline that publishes images or wheels should set it**, so published images keep
their full contents and one shared cached stage.
An explicit `--build-args WITH_X:0|1` still overrides the computed value for that library.

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
