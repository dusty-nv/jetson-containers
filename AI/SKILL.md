---
name: jetson-containers
description: Build, configure, and extend the jetson-containers modular Docker build system for AI/ML on NVIDIA Jetson and ARM/x86 platforms. Use when working with packages, Dockerfiles, config files, build commands, or adding new container packages to this repo.
---

# jetson-containers

Modular container build system for AI/ML on NVIDIA Jetson (tegra-aarch64), SBSA (aarch64), and x86_64. Chains package Dockerfiles into multi-stage Docker images with version-aware dependency resolution across JetPack/L4T, CUDA, Python, and Ubuntu.

## Platform Architecture

Starting from JetPack 7.2, **all devices use SBSA (Server Base System Architecture) with CUDA 13.2**. This unifies Jetson and server ARM under a single architecture, eliminating the previous tegra-aarch64 vs SBSA split.

| JetPack | L4T | Architecture | CUDA |
|---------|-----|--------------|------|
| 6.x | r36.x | tegra-aarch64 (Jetson) / SBSA (server ARM) | 12.x |
| 7.0-7.1 | r38.x | tegra-aarch64 (Jetson) / SBSA (server ARM) | 13.0-13.1 |
| **7.2+** | r38.x+ | **SBSA for all devices** | **13.2** |

This means packages targeting JetPack 7.2+ no longer need to differentiate between Jetson and server ARM — `IS_SBSA` is effectively true for all ARM devices.

## Project Layout

```
jetson-containers          # CLI launcher (bash)
jetson_containers/         # Core Python package
  packages.py              # Package discovery, config loading, dependency resolution
  container.py             # build_container, find_container, test_container
  l4t_version.py           # L4T/CUDA/JetPack/Python detection; system variables
  build.py                 # Build CLI entry point
  tag.py                   # autotag logic
  utils.py, logging.py, network.py, ...
packages/                  # 330+ package definitions
  ml/                      # PyTorch, TensorFlow, JAX, ONNX
  llm/                     # vLLM, SGLang, transformers, ollama
  vlm/                     # LLaVA, VILA
  cv/                      # OpenCV, DeepStream, diffusion
  physicalAI/              # ROS, LeRobot, Isaac Sim
  cuda/, speech/, agents/, vit/, rag/, net/, hw/, ...
```

## CLI Commands

```bash
jetson-containers build pytorch                    # single container
jetson-containers build pytorch jupyterlab         # chained packages
jetson-containers build --multiple pytorch tensorflow  # separate containers
jetson-containers build --name=my_container pytorch
jetson-containers build --base=xyz:latest pytorch  # custom base image
jetson-containers build --simulate pytorch         # dry run
jetson-containers build --skip-tests=all pytorch
jetson-containers build --list-packages
jetson-containers build --show-packages ros*
jetson-containers run $(autotag pytorch)           # find + run compatible image
```

### Version overrides via environment variables

```bash
LSB_RELEASE=24.04 CUDA_VERSION=13.1 PYTORCH_VERSION=2.8 jetson-containers build vllm
```

| Variable | Controls |
|----------|----------|
| `CUDA_VERSION` | CUDA toolkit version |
| `CUDNN_VERSION` | cuDNN version |
| `TENSORRT_VERSION` | TensorRT version |
| `PYTHON_VERSION` | Python version |
| `PYTORCH_VERSION` | PyTorch version |
| `LSB_RELEASE` | Ubuntu version (22.04, 24.04) |
| `CUDA_ARCH` | GPU compute architectures |

## Package System

A package is a directory under `packages/` containing a Dockerfile and optional config files.

### Dockerfile YAML header (primary format)

```dockerfile
#---
# name: my-package
# alias: my-pkg
# group: ml
# config: config.py
# depends: [python, numpy, onnx]
# requires: '>=36'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
...
```

### Package metadata keys

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Package name |
| `alias` | `str \| list[str]` | Alternate names |
| `group` | `str` | Category (ml, llm, cuda, etc.) |
| `depends` | `str \| list[str]` | Build dependencies |
| `requires` | `str \| list[str]` | L4T/CUDA/Python version constraints (e.g. `>=36`, `>=cu124`, `>=py310`) |
| `config` | `str \| list[str]` | Config files to load (.py, .json, .yml) |
| `build_args` | `dict` | Docker `--build-arg` key/value pairs |
| `build_flags` | `str` | Extra `docker build` flags |
| `test` | `str \| list[str]` | Test scripts (.py, .sh, or shell command) |
| `disabled` | `bool` | Disable the package |
| `dockerfile` | `str` | Dockerfile filename |
| `prefix` / `postfix` | `str` | Container tag prefix/postfix |
| `docs` / `notes` | `str` | Documentation text |

### Config formats

**config.py** - Dynamic configuration using `package` dict injected by the build system:

```python
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

pkg = package.copy()
pkg['name'] = f'my-package:{version}'
pkg['build_args'] = {
    'MY_VERSION': version,
    'CUDA_ARCHITECTURES': ';'.join(CUDA_ARCHITECTURES),
}
if L4T_VERSION.major >= 36:
    pkg['requires'] = '>=36'

package = pkg  # or list of packages
```

**config.json** - Static config, commonly for meta-packages:

```json
{
    "l4t-pytorch": {
        "group": "ml",
        "depends": ["pytorch", "torchvision", "torchaudio", "torch2trt", "opencv", "pycuda"]
    }
}
```

**config.yaml** - Same keys as Dockerfile header, in standalone YAML.

### System variables available in config.py

Import from `jetson_containers`:

| Variable | Type | Description |
|----------|------|-------------|
| `L4T_VERSION` | `packaging.version.Version` | L4T version |
| `JETPACK_VERSION` | `packaging.version.Version` | JetPack version |
| `CUDA_VERSION` | `packaging.version.Version` | CUDA version |
| `PYTHON_VERSION` | `packaging.version.Version` | Python version |
| `CUDA_ARCHITECTURES` | `list[int]` | GPU arch codes (e.g. `[72, 87, 101]`) |
| `LSB_RELEASE` | `str` | Ubuntu version (`22.04`, `24.04`) |
| `IS_SBSA` | `bool` | True for server ARM (GH200/GB200) |
| `SYSTEM_ARM` | `bool` | True for any ARM (Jetson or SBSA) |

### Cross-package imports in config.py

```python
from packages.ml.pytorch.version import PYTORCH_VERSION
```

Resolved by `_PackagesFinder` meta path finder - no `__init__.py` needed.

## Adding a New Package

1. Create `packages/<group>/<name>/` directory
2. Add a `Dockerfile` with YAML header (`#---` block)
3. Optionally add `config.py` for dynamic version logic
4. Optionally add `test.py` or `test.sh`
5. Verify: `jetson-containers build --show-packages <name>`
6. Build: `jetson-containers build <name>`

### Minimal example

```
packages/ml/my-package/
├── Dockerfile
└── test.py        # optional
```

```dockerfile
#---
# name: my-package
# group: ml
# depends: [python, numpy]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip3 install my-package
```

### Versioned package example (with config.py)

```
packages/ml/my-package/
├── Dockerfile
├── config.py
└── test.py
```

The `config.py` returns a list of package dicts (one per version), each with a versioned `name` like `my-package:1.0`. Use helper functions like `package.copy()` to create variants, set `build_args` based on `L4T_VERSION` / `CUDA_VERSION`, and set `requires` to constrain which platforms each version supports.

### Meta-package (no Dockerfile)

Create a JSON config that only declares dependencies:

```json
{
    "my-meta-package": {
        "group": "ml",
        "depends": ["pytorch", "torchvision", "jupyterlab"]
    }
}
```

## Container Tags

Format: `r{L4T}.{arch}-cu{CUDA}-{LSB}` (e.g. `r36.4.0-cu126-22.04`)

Registry: `dustynv/<package>:<version>-<tag>` on DockerHub.

## Code Style

- **black** for Python formatting
- **flake8** for linting
- **pre-commit** hooks configured in `.pre-commit-config.yaml`

## Additional Resources

- [docs/build.md](docs/build.md) - Build usage and version control
- [docs/packages.md](docs/packages.md) - Package definition format
- [docs/run.md](docs/run.md) - Running containers and autotag
- [docs/setup.md](docs/setup.md) - System setup
- [docs/code-style.md](docs/code-style.md) - Code style guide
