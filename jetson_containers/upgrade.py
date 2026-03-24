#!/usr/bin/env python3
"""
Check packages for available upstream version upgrades, and optionally apply them.

Usage:
    jetson-containers upgrade [PACKAGE ...] [--apply]
    python3 -m jetson_containers.upgrade [PACKAGE ...]  [--apply]

For every tracked package this script:
  1. Extracts the current pinned version from config.py / version.py.
  2. Fetches the latest version from upstream (PyPI JSON API or GitHub).
     For GitHub sources it checks *both* the latest release tag and the
     version field in pyproject.toml / setup.cfg / setup.py on the default
     branch, and returns whichever is higher — because many repos update
     their metadata files between tagged releases.
  3. Reports packages where an upgrade is available.
  4. With --apply, rewrites the default=True entry in config.py.
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from packaging.version import Version, InvalidVersion
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Registry
# key  : path relative to packages/  (used to locate config.py)
# value: dict with one of:
#   {'source': 'pypi',   'name': '<pypi-package-name>'}
#   {'source': 'github', 'repo': 'owner/repo', 'strip': '<tag-prefix>'}
#   {'source': 'skip',   'reason': '...'}
#
# strip: prefix to remove from a GitHub tag to get a bare version string.
#   'v'  → v0.18.0   → 0.18.0
#   'n'  → n8.0.1    → 8.0.1   (FFmpeg)
#   'b'  → b5000     → 5000    (llama.cpp — non-semver, compared as string)
#   ''   → 4.14.0    → 4.14.0  (no prefix)
# ---------------------------------------------------------------------------
REGISTRY = {

    # ── LLM ──────────────────────────────────────────────────────────────────
    'llm/awq':                              {'source': 'github', 'repo': 'mit-han-lab/llm-awq',                     'strip': 'v'},
    'llm/bitsandbytes':                     {'source': 'pypi',   'name': 'bitsandbytes'},
    'llm/deepspeed':                        {'source': 'pypi',   'name': 'deepspeed'},
    'llm/deepspeed/deepspeed-kernels':      {'source': 'pypi',   'name': 'deepspeed-kernels'},
    'llm/dynamo/dynamo':                    {'source': 'pypi',   'name': 'ai-dynamo'},
    'llm/dynamo/kai-scheduler':             {'source': 'github', 'repo': 'NVIDIA/KAI-Scheduler',                    'strip': 'v'},
    'llm/dynamo/mooncake':                  {'source': 'pypi',   'name': 'mooncake-transfer-engine'},
    'llm/dynamo/nixl':                      {'source': 'pypi',   'name': 'nixl'},
    'llm/exllama':                          {'source': 'pypi',   'name': 'exllamav3'},
    'llm/gptqmodel':                        {'source': 'pypi',   'name': 'gptqmodel'},
    'llm/ktransformers':                    {'source': 'pypi',   'name': 'ktransformers'},
    'llm/llama_cpp':                        {'source': 'github', 'repo': 'ggerganov/llama.cpp',                     'strip': 'b'},
    'llm/minference':                       {'source': 'pypi',   'name': 'minference'},
    'llm/mlc':                              {'source': 'github', 'repo': 'mlc-ai/mlc-llm',                         'strip': 'v'},
    'llm/nemo':                             {'source': 'pypi',   'name': 'nemo_toolkit'},
    'llm/ollama':                           {'source': 'github', 'repo': 'ollama/ollama',                           'strip': 'v'},
    'llm/open-webui':                       {'source': 'github', 'repo': 'open-webui/open-webui',                   'strip': 'v'},
    'llm/optimum':                          {'source': 'pypi',   'name': 'optimum'},
    'llm/sglang':                           {'source': 'pypi',   'name': 'sglang'},
    'llm/tensorrt_optimizer/nvidia-modelopt': {'source': 'pypi', 'name': 'nvidia-modelopt'},
    'llm/tensorrt_optimizer/tensorrt_llm':  {'source': 'github', 'repo': 'NVIDIA/TensorRT-LLM',                    'strip': 'v'},
    'llm/text-generation-inference':        {'source': 'github', 'repo': 'huggingface/text-generation-inference',  'strip': 'v'},
    'llm/text-generation-webui':            {'source': 'github', 'repo': 'oobabooga/text-generation-webui',        'strip': 'v'},
    'llm/transformers':                     {'source': 'pypi',   'name': 'transformers'},
    'llm/unsloth':                          {'source': 'pypi',   'name': 'unsloth'},
    'llm/vllm':                             {'source': 'pypi',   'name': 'vllm'},
    'llm/xgrammar':                         {'source': 'pypi',   'name': 'xgrammar'},

    # ── Attention ─────────────────────────────────────────────────────────────
    'attention/ParaAttention':              {'source': 'pypi',   'name': 'para-attn'},
    'attention/block-sparse-attention':     {'source': 'github', 'repo': 'mit-han-lab/Block-Sparse-Attention',   'strip': 'v'},
    'attention/flash-attention':            {'source': 'github', 'repo': 'Dao-AILab/flash-attention',              'strip': 'v'},
    'attention/flashinfer':                 {'source': 'pypi',   'name': 'flashinfer-python'},
    'attention/flexprefill':                {'source': 'github', 'repo': 'ByteDance-Seed/FlexPrefill',            'strip': 'v'},
    'attention/huggingface_kernels':        {'source': 'pypi',   'name': 'kernels'},
    'attention/jvp-flash-attention':        {'source': 'pypi',   'name': 'jvp-flash-attention'},
    'attention/log-linear-attention':       {'source': 'github', 'repo': 'HanGuo97/log-linear-attention',         'strip': 'v'},
    'attention/radial-attention':           {'source': 'github', 'repo': 'mit-han-lab/radial-attention',         'strip': 'v'},
    'attention/sage-attention':             {'source': 'pypi',   'name': 'sageattention'},
    'attention/sparge-attention':           {'source': 'github', 'repo': 'thu-ml/SpargeAttn',                    'strip': 'v'},
    'attention/tilelang':                   {'source': 'pypi',   'name': 'tilelang'},
    'attention/xattention':                 {'source': 'pypi',   'name': 'xattn'},
    'attention/xformers':                   {'source': 'github', 'repo': 'facebookresearch/xformers',              'strip': 'v'},

    # ── ML ────────────────────────────────────────────────────────────────────
    'ml/ctranslate2':                       {'source': 'pypi',   'name': 'ctranslate2'},
    'ml/jax':                               {'source': 'pypi',   'name': 'jax'},
    'ml/mamba/causalconv1d':                {'source': 'pypi',   'name': 'causal-conv1d'},
    'ml/mamba/mamba':                       {'source': 'pypi',   'name': 'mamba-ssm'},
    'ml/numeric/cupy':                      {'source': 'pypi',   'name': 'cupy'},
    'ml/numeric/numpy':                     {'source': 'pypi',   'name': 'numpy'},
    'ml/numeric/warp':                      {'source': 'pypi',   'name': 'warp-lang'},
    'ml/onnxruntime':                       {'source': 'github', 'repo': 'microsoft/onnxruntime',                  'strip': 'v'},
    'ml/pytorch':                           {'source': 'github', 'repo': 'pytorch/pytorch',                        'strip': 'v'},
    'ml/pytorch/torchaudio':                {'source': 'github', 'repo': 'pytorch/audio',                          'strip': 'v'},
    'ml/pytorch/torchao':                   {'source': 'github', 'repo': 'pytorch/ao',                             'strip': 'v'},
    'ml/pytorch/torchvision':               {'source': 'github', 'repo': 'pytorch/vision',                         'strip': 'v'},
    'ml/rapids/cudf':                       {'source': 'github', 'repo': 'rapidsai/cudf',                          'strip': 'v'},
    'ml/rapids/cuml':                       {'source': 'github', 'repo': 'rapidsai/cuml',                          'strip': 'v'},
    'ml/tensorflow':                        {'source': 'skip',   'reason': 'prebuilt NVIDIA wheels tied to JetPack'},
    'ml/triton':                            {'source': 'github', 'repo': 'triton-lang/triton',                     'strip': 'v'},

    # ── Computer Vision ───────────────────────────────────────────────────────
    'cv/cv-cuda':                           {'source': 'github', 'repo': 'CVCUDA/CV-CUDA',                         'strip': 'v'},
    'cv/diffusion/diffusers':               {'source': 'pypi',   'name': 'diffusers'},
    'cv/opencv':                            {'source': 'github', 'repo': 'opencv/opencv',                          'strip': ''},

    # ── CUDA ──────────────────────────────────────────────────────────────────
    'cuda/cuda-python':                     {'source': 'pypi',   'name': 'cuda-python'},
    'cuda/cutlass':                         {'source': 'github', 'repo': 'NVIDIA/cutlass',                         'strip': 'v'},
    'cuda/pycuda':                          {'source': 'pypi',   'name': 'pycuda'},

    # ── Multimedia ────────────────────────────────────────────────────────────
    'multimedia/decord':                    {'source': 'pypi',   'name': 'decord2'},
    'multimedia/ffmpeg':                    {'source': 'github', 'repo': 'FFmpeg/FFmpeg',                          'strip': 'n'},

    # ── Code / Tools ──────────────────────────────────────────────────────────
    'code/jupyterlab':                      {'source': 'pypi',   'name': 'jupyterlab'},

    # ── Speech ────────────────────────────────────────────────────────────────
    'speech/faster-whisper':                {'source': 'pypi',   'name': 'faster-whisper'},
    'speech/piper1-tts':                    {'source': 'pypi',   'name': 'piper-tts'},

    # ── Physical AI / ROS ─────────────────────────────────────────────────────
    'physicalAI/ros':                       {'source': 'skip',   'reason': 'version depends on Ubuntu/ROS distro mapping'},
}


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def _parse_ver(v: str) -> Optional['Version']:
    if not HAS_PACKAGING:
        return None
    try:
        return Version(str(v).strip())
    except (InvalidVersion, TypeError):
        return None


def _higher(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Return whichever version string is higher; fall back to string compare."""
    if not a:
        return b
    if not b:
        return a
    va, vb = _parse_ver(a), _parse_ver(b)
    if va and vb:
        return str(max(va, vb))
    return a if a >= b else b  # string fallback (e.g. llama.cpp 'b' tags)


def _is_newer(latest: str, current: str) -> bool:
    lv, cv = _parse_ver(latest), _parse_ver(current)
    if lv and cv:
        return lv > cv
    return latest != current and latest > current  # string fallback


# ---------------------------------------------------------------------------
# Upstream version fetching
# ---------------------------------------------------------------------------

def _http_get(url: str, json: bool = False, retries: int = 3, backoff: int = 3,
              token: str = None):
    """Minimal HTTP GET that works standalone (only needs `requests`)."""
    if not HAS_REQUESTS:
        return None
    headers = {'User-Agent': 'jetson-containers-upgrade/1.0'}
    if token:
        headers['Authorization'] = f'token {token}'
    for attempt in range(retries):
        try:
            r = _requests.get(url, headers=headers, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json() if json else r.text
        except Exception:
            if attempt < retries - 1:
                time.sleep(backoff)
    return None


def _pypi_latest(pkg_name: str) -> Optional[str]:
    # Try the jetson_containers helper first (handles caching); fall back to direct call.
    try:
        from .pypi_utils import get_latest_version
        return get_latest_version(pkg_name)
    except Exception:
        pass
    data = _http_get(f"https://pypi.org/pypi/{pkg_name}/json", json=True)
    if not data or 'releases' not in data:
        return None
    if not HAS_PACKAGING:
        return data.get('info', {}).get('version')
    versions = sorted(data['releases'].keys(), key=lambda v: _parse_ver(v) or Version('0'))
    return versions[-1] if versions else None


def _github_latest_tag(repo: str, strip: str) -> Optional[str]:
    import os
    token = os.environ.get('GITHUB_TOKEN')
    data = _http_get(f"https://api.github.com/repos/{repo}/tags", json=True, token=token)
    if not data:
        return None
    tag = data[0].get('name', '') if data else ''
    if strip and tag.startswith(strip):
        tag = tag[len(strip):]
    return tag or None


_VERSION_LIKE = re.compile(r'^\d[\d.]*(?:[-+a-zA-Z0-9.]*)?$')


def _looks_like_version(v: str) -> bool:
    """Return True if v looks like a real version, not a template or git ref."""
    if not v or '{' in v or len(v) > 30:
        return False
    return bool(_VERSION_LIKE.match(v.strip()))


def _github_pyproject_version(repo: str) -> Optional[str]:
    """
    Check pyproject.toml → setup.cfg → setup.py on main/master for a
    version field.  Many projects bump these between tagged releases,
    so this can return a version higher than the latest tag.
    """
    for branch in ('main', 'master'):
        for filename, pattern in (
            ('pyproject.toml', r'^\s*version\s*=\s*["\']([^"\']+)["\']'),
            ('setup.cfg',      r'^\s*version\s*=\s*(\S+)'),
            ('setup.py',       r'version\s*=\s*["\']([^"\']+)["\']'),
        ):
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{filename}"
            content = _http_get(url)
            if content:
                m = re.search(pattern, content, re.MULTILINE)
                if m:
                    v = m.group(1).strip()
                    if _looks_like_version(v):
                        return v
    return None


def _fetch_latest(info: dict) -> Optional[str]:
    source = info['source']

    if source == 'skip':
        return None

    if source == 'pypi':
        return _pypi_latest(info['name'])

    if source == 'github':
        repo = info['repo']
        strip = info.get('strip', '')

        tag_ver = _github_latest_tag(repo, strip)
        meta_ver = _github_pyproject_version(repo)

        # Return whichever is higher (metadata often leads tags)
        return _higher(tag_ver, meta_ver)

    return None


# ---------------------------------------------------------------------------
# Current-version extraction from config.py / version.py
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(
    r"^(\s*)(\w[\w-]*)\s*\(\s*['\"](\d[\d.\w-]*)['\"]",
    re.MULTILINE,
)
_DEFAULT_ENTRY_RE = re.compile(
    r"^(\s*)(\w[\w-]*)\s*\(\s*['\"](\d[\d.\w-]*)['\"]((?:[^()]|\([^()]*\))*?)default\s*=\s*True",
    re.MULTILINE | re.DOTALL,
)


def _extract_current_version(config_path: Path) -> Optional[str]:
    """
    Return the current pinned version for a package.

    Detection order:
    1. version.py: finds all  SOMETHING = Version('X.Y')  assignments and
       returns the highest value (these files gate on hardware, so the highest
       is the most-current target).
    2. config.py package list: entry with default=True.
    3. Highest semantic version found in the package list entries.
    """
    pkg_dir = config_path.parent

    # 1. version.py
    version_py = pkg_dir / 'version.py'
    if version_py.exists():
        vtext = version_py.read_text()
        raw = re.findall(
            r'^\s*\w+\s*=\s*Version\s*\(\s*[\'"](\d[\d.]+)[\'"]',
            vtext, re.MULTILINE,
        )
        parsed = [_parse_ver(v) for v in raw if _parse_ver(v)]
        if parsed:
            return str(max(parsed))

    text = config_path.read_text()

    # 2. Entry with default=True
    pkg_block = re.search(r'package\s*=\s*\[', text)
    if not pkg_block:
        return None
    block = text[pkg_block.start():]

    m = _DEFAULT_ENTRY_RE.search(block)
    if m:
        return m.group(3)

    # 3. Highest semver in the package list
    matches = list(_ENTRY_RE.finditer(block))
    if not matches:
        return None
    parsed = [((_parse_ver(e.group(3)), e.group(3))) for e in matches if _parse_ver(e.group(3))]
    if parsed:
        return max(parsed, key=lambda x: x[0])[1]
    return matches[-1].group(3)


# ---------------------------------------------------------------------------
# Apply upgrade: rewrite config.py
# ---------------------------------------------------------------------------

def _apply_upgrade(config_path: Path, current: str, latest: str) -> bool:
    """
    Replace the version string in the default=True entry with `latest`.
    Returns True on success.
    """
    text = config_path.read_text()
    pkg_block = re.search(r'package\s*=\s*\[', text)
    if not pkg_block:
        _warn(f"  No package list in {config_path}")
        return False

    prefix, block = text[:pkg_block.start()], text[pkg_block.start():]

    m = _DEFAULT_ENTRY_RE.search(block)
    if not m:
        _warn(f"  No default=True entry found in {config_path}")
        return False

    old_fragment = block[m.start():m.end()]
    new_fragment = re.sub(
        r'''(['"])''' + re.escape(current) + r'''\1''',
        lambda mo: mo.group(1) + latest + mo.group(1),
        old_fragment,
        count=1,
    )
    if new_fragment == old_fragment:
        _warn(f"  Version '{current}' not replaced in {config_path}")
        return False

    config_path.write_text(prefix + block[:m.start()] + new_fragment + block[m.end():])
    return True


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _ok(msg):   print(f"  \033[32m✓\033[0m {msg}")
def _up(msg):   print(f"  \033[33m↑\033[0m {msg}")
def _skip(msg): print(f"  \033[90m–\033[0m {msg}")
def _warn(msg): print(f"  \033[31m!\033[0m {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Check/upgrade package versions from PyPI and GitHub.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('packages', nargs='*', metavar='PACKAGE',
        help='Packages to check (registry key or short name). Default: all.')
    parser.add_argument('--apply', action='store_true',
        help='Write updated version into config.py (default: report only).')
    parser.add_argument('--show-registry', action='store_true',
        help='List all tracked packages and exit.')
    args = parser.parse_args(argv)

    if args.show_registry:
        print(f'\n{"Package":<50} {"Source":<8} {"Upstream"}\n' + '─' * 80)
        for key, info in sorted(REGISTRY.items()):
            src = info['source']
            detail = info.get('repo', '') or info.get('name', '') or info.get('reason', '')
            print(f'  {key:<48} [{src}]  {detail}')
        print()
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    packages_root = repo_root / 'packages'

    # Filter registry by user-supplied names
    if args.packages:
        targets = {}
        for pkg in args.packages:
            if pkg in REGISTRY:
                targets[pkg] = REGISTRY[pkg]
            else:
                matches = [k for k in REGISTRY if k.endswith('/' + pkg) or k == pkg]
                if matches:
                    for k in matches:
                        targets[k] = REGISTRY[k]
                else:
                    _warn(f"'{pkg}' not found in registry (run --show-registry to see all)")
    else:
        targets = REGISTRY

    print(f"\n{'Package':<50} {'Current':<12} {'Latest':<12} {'Status'}")
    print('─' * 95)

    upgraded, available, errors = [], [], []

    for key, info in sorted(targets.items()):
        config_path = packages_root / key / 'config.py'

        if info['source'] == 'skip':
            _skip(f"{key:<48} — {info.get('reason', 'skipped')}")
            continue

        if not config_path.exists():
            _skip(f"{key:<48} — config.py not found")
            continue

        current = _extract_current_version(config_path)
        if not current:
            _skip(f"{key:<48} — version not detected")
            continue

        latest = _fetch_latest(info)
        if not latest:
            _warn(f"{key:<48} — upstream lookup failed")
            errors.append(key)
            continue

        if not _is_newer(latest, current):
            _ok(f"{key:<48} {current:<12} {latest:<12} up-to-date")
            continue

        upstream = info.get('repo', '') or info.get('name', '')
        _up(f"{key:<48} {current:<12} {latest:<12} upgrade available  [{upstream}]")
        available.append((key, current, latest, config_path))

        if args.apply:
            ok = _apply_upgrade(config_path, current, latest)
            if ok:
                print(f"       \033[32mapplied\033[0m → {config_path.relative_to(repo_root)}")
                upgraded.append(key)
            else:
                errors.append(key)

    print()

    if args.apply:
        if upgraded:
            print(f"Upgraded {len(upgraded)} package(s): {', '.join(upgraded)}")
            print("Review diffs, adjust JetPack `requires` if needed, then open a PR to `dev`.")
        else:
            print("No packages were modified.")
    else:
        if available:
            print(f"{len(available)} upgrade(s) available. Re-run with --apply to update config.py files.")
        else:
            print("All tracked packages are up-to-date.")

    if errors:
        print(f"\n{len(errors)} error(s): {', '.join(errors)}", file=sys.stderr)

    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
