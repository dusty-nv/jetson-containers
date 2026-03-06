---
name: upgrade-packages
description: Upgrade jetson-containers packages to their latest stable versions. Use when the user asks to update, upgrade, or bump a package version, or wants to add a new version of an existing package in config.py, Dockerfiles, or version.py.
---

# Upgrade Packages

Workflow for upgrading jetson-containers packages to their latest stable release. Each package can pin versions in multiple places — all must be updated together.

## Upgrade Checklist

For any package upgrade, work through these steps:

1. **Find the latest stable version** upstream (GitHub releases, PyPI, official docs)
2. **Identify all version-pinning locations** for the package (see patterns below)
3. **Update config.py** — add new version entry or update existing one
4. **Update version.py** — change default version mapping if present
5. **Update Dockerfiles / install scripts** — only if they hardcode versions
6. **Update dependent packages** — cascade version changes (e.g. PyTorch → torchvision)
7. **Verify** — `jetson-containers build --show-packages <name>` then `--simulate`

## Where Versions Live

| File | What to change |
|------|----------------|
| `config.py` | Version entries in the `package` list — this is always the primary file |
| `version.py` | Default version selection based on L4T/CUDA — update the mapping |
| `Dockerfile` | Usually version-agnostic (uses `ARG`), rarely needs changes |
| `install.sh` / `build.sh` | Usually version-agnostic (uses `$VAR`), rarely needs changes |
| `config.json` / `config.yaml` | Static meta-packages — update `depends` if pinning versioned deps |

## Version Pinning Patterns

### Pattern 1: Pip install (most common)

Versions flow through `config.py` → `build_args` → `ARG` in Dockerfile → `install.sh`.

**config.py** — add a new entry to the package list:

```python
package = [
    my_package('1.2.0', requires='>=36', default=False),
    my_package('1.3.0', requires='>=36', default=True),   # new version
]
```

Set `default=True` on the new version (and `False` on the old). The helper function typically does:

```python
def my_package(version, requires=None, default=True):
    pkg = package.copy()
    pkg['name'] = f'my-package:{version}'
    pkg['build_args'] = {'MY_PACKAGE_VERSION': version}
    if requires:
        pkg['requires'] = requires
    return pkg
```

**install.sh** — no change needed if it already uses the variable:

```bash
uv pip install my-package==${MY_PACKAGE_VERSION}
```

### Pattern 2: Git clone with tag/branch

```python
def vllm(version, branch=None, ...):
    pkg['build_args'] = {
        'VLLM_VERSION': version,
        'VLLM_BRANCH': branch or f'v{version}',
    }
```

**build.sh** uses the branch arg:

```bash
git clone --branch=${VLLM_BRANCH} --recursive --depth=1 https://github.com/org/repo /opt/repo
```

To upgrade: add a new `vllm('0.17.0')` entry. The branch defaults to `v0.17.0`.

### Pattern 3: Wget / prebuilt wheel URLs

Some packages download prebuilt wheels by URL. These require finding the new URL.

```python
pytorch_wget('1.10', 'torch-1.10.0-cp36-cp36m-linux_aarch64.whl',
    'https://nvidia.box.com/shared/static/xyz.whl', '==32.*')
```

Or URL lookup dicts (e.g. TensorFlow):

```python
prebuilt_wheels = {
    ('36', '2.16.1', 'tf2'): ('https://...whl', 'tensorflow-2.16.1+nv24.06-....whl'),
}
```

To upgrade: add a new entry with the new URL. Check the upstream release page for the download link.

### Pattern 4: CUDA deb packages

```python
cuda_package('13.2', 'https://developer.download.nvidia.com/.../cuda-repo-..._13.2.0-..._arm64.deb', ...)
```

URLs follow NVIDIA's pattern. Check https://developer.nvidia.com/cuda-downloads for new releases.

## Updating version.py

Most major packages have a `version.py` that selects the default version based on the platform. After adding a new version in `config.py`, update the mapping:

```python
# Before
if CUDA_VERSION >= Version('13.0'):
    PYTORCH_VERSION = Version('2.10')

# After
if CUDA_VERSION >= Version('13.2'):
    PYTORCH_VERSION = Version('2.12')
elif CUDA_VERSION >= Version('13.0'):
    PYTORCH_VERSION = Version('2.10')
```

The `version.py` pattern is:

1. Check env var override first (`os.environ.get('PYTORCH_VERSION')`)
2. Then select based on `L4T_VERSION` / `CUDA_VERSION` / `LSB_RELEASE`
3. Other packages import this: `from packages.ml.pytorch.version import PYTORCH_VERSION`

## Cascading Dependencies

When upgrading a core package, dependent packages often need matching updates.

### PyTorch ecosystem

Upgrading PyTorch requires updating its companions:

| Package | Config location | Dependency |
|---------|-----------------|------------|
| `torchvision` | `packages/ml/pytorch/torchvision/config.py` | `pytorch='{version}'` param |
| `torchaudio` | `packages/ml/pytorch/torchaudio/config.py` | `pytorch='{version}'` param |
| `torchao` | `packages/ml/pytorch/torchao/config.py` | `pytorch='{version}'` param |
| `torch2trt` | `packages/ml/pytorch/torch2trt/config.py` | depends on pytorch |

Each companion config ties versions to PyTorch:

```python
torchvision('0.27.0', pytorch='2.12', requires='>=36'),
```

When adding PyTorch 2.13, also add matching torchvision, torchaudio, torchao entries. Check [PyTorch's compatibility matrix](https://github.com/pytorch/pytorch/wiki/PyTorch-Versions) for the right companion versions.

### CUDA ecosystem

Upgrading CUDA may require updating:
- `cudnn` — `packages/cuda/cudnn/config.py`
- `tensorrt` — `packages/tensorrt/config.py`
- `cutlass` — `packages/cuda/cutlass/config.py`

### LLM stack

Upgrading vLLM or SGLang may require matching:
- `flashinfer` — `packages/attention/flashinfer/config.py`
- `flash-attention` — `packages/attention/flash-attention/config.py`
- `xformers` — `packages/attention/xformers/config.py`

## Using `update_dependencies()`

When a config needs to override a transitive dependency version:

```python
from jetson_containers import update_dependencies

pkg['depends'] = update_dependencies(pkg['depends'], f"pytorch:{pytorch_version}")
```

This replaces `pytorch` with `pytorch:2.12` in the dependency list without duplicating it.

## Finding Latest Stable Versions

Use web search or check these sources:

| Source | URL pattern |
|--------|-------------|
| PyPI | `https://pypi.org/project/<package>/` |
| GitHub releases | `https://github.com/<org>/<repo>/releases/latest` |
| NVIDIA CUDA | `https://developer.nvidia.com/cuda-downloads` |
| PyTorch | `https://pytorch.org/get-started/locally/` |

Some packages use `github_latest_tag()` from `jetson_containers` to auto-detect:

```python
from jetson_containers import github_latest_tag
version = github_latest_tag('org/repo', prefix='v')
```

## Verification

After making changes:

```bash
jetson-containers build --show-packages <name>     # confirm metadata is correct
jetson-containers build --simulate <name>           # dry-run the build chain
jetson-containers build <name>                      # actual build + tests
```

Check that:
- The new version appears in `--show-packages`
- The `default` flag is on the correct version
- `requires` constraints are appropriate for the target platforms
- Dependent packages resolve to compatible versions

## Example: Full Upgrade of a Pip Package

Upgrading `faster-whisper` from 1.2.1 to 1.3.0:

1. Edit `packages/speech/faster-whisper/config.py`:
   - Add `faster_whisper("1.3.0", default=True)`
   - Set old entry to `default=False`

2. Check `version.py` (if it exists) and update default mapping

3. Run `jetson-containers build --show-packages faster-whisper` to verify

4. No Dockerfile/install.sh changes needed — they use `${FASTER_WHISPER_VERSION}`
