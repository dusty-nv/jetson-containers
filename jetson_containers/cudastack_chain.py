#!/usr/bin/env python3
"""
cudastack_chain.py — pure chain -> cudastack WITH_* set + stage-tag suffix.

This module is intentionally stdlib-only and does NOT import from
packages.py or container.py: every function here takes plain dicts/lists so
it can be tested standalone (tests/test_cudastack_chain.py) without pulling
in the rest of jetson_containers (which has third-party runtime deps and
network/pip side effects at import time via utils.check_dependencies()).

Background (packages/cuda/cudastack/config.py:160-168): a cudastack preset
(e.g. 'minimal', 'standard') is a dict of WITH_* build args ('0'/'1' strings)
telling the cudastack Dockerfile which CUDA libraries to install in its one
consolidated RUN. WITH_CUDNN is always '1' in every preset today.

This module answers: given the packages ABOVE cudastack in a resolved build
chain (each optionally declaring a `cuda_libs` list of library names it
needs) and the preset for the cudastack variant being built, which WITH_*
flags does this specific chain actually need turned on (narrowed within what
the preset allows), and what Docker stage-tag suffix represents that set.
"""

from __future__ import annotations

import hashlib

# 1:1 vocabulary: cuda_libs name -> WITH_* build-arg name.
CUDA_LIBS = {
    'cudnn': 'WITH_CUDNN',
    'tensorrt': 'WITH_TENSORRT',
    'nccl': 'WITH_NCCL',
    'cudss': 'WITH_CUDSS',
    'cusparselt': 'WITH_CUSPARSELT',
    'cutensor': 'WITH_CUTENSOR',
    'nvshmem': 'WITH_NVSHMEM',
    'nvpl': 'WITH_NVPL',
    'gdrcopy': 'WITH_GDRCOPY',
}

# cuDNN is always installed by cudastack today (config.py:160: WITH_CUDNN is
# hard-coded '1' in every preset), regardless of what any package declares.
ALWAYS_ON_LIBS = frozenset({'cudnn'})

_STAGE_PREFIX = 'cudastack_'


def compute_set(chain_above, preset):
    """
    Compute the WITH_* dict to actually use for a cudastack build, given the
    packages above cudastack in the resolved chain and the preset for the
    chosen cudastack variant.

    Args:
        chain_above: list of package dicts (as found in the resolved build
            chain, above cudastack). Each package dict may declare a
            `cuda_libs` list of library names (keys of CUDA_LIBS) it needs.
            A package dict with NO `cuda_libs` key at all is "undeclared"
            and is treated as needing everything the preset offers. A
            package dict with `cuda_libs: []` is declared, and needs
            nothing on its own.
        preset: dict of WITH_* -> '0'/'1' (e.g. the output of
            cuda_stack_args() + the preset toggles in packages/cuda/
            cudastack/config.py:160-168).

    Returns:
        A new dict WITH_* -> '0'/'1'. Always a subset of `preset` (narrow-
        only): a WITH_* that is '0' in the preset is never turned '1' here,
        even if some package declared the corresponding lib.

    An empty chain_above returns a copy of the preset unchanged (there is
    nothing to narrow against).
    """
    preset = dict(preset)

    if not chain_above:
        return preset

    requested = set(ALWAYS_ON_LIBS)
    any_undeclared = False

    for pkg in chain_above:
        if 'cuda_libs' not in pkg:
            any_undeclared = True
            continue
        requested.update(pkg['cuda_libs'] or [])

    if any_undeclared:
        # An undeclared package contributes the full preset: everything the
        # preset allows passes through unchanged.
        requested.update(CUDA_LIBS.keys())

    result = {}
    for lib, with_key in CUDA_LIBS.items():
        preset_val = preset.get(with_key, '0')
        enabled = lib in requested and preset_val == '1'
        result[with_key] = '1' if enabled else '0'

    # Preserve any WITH_* (or other) keys the preset carries that aren't in
    # CUDA_LIBS's vocabulary, unchanged, so callers that pass a superset
    # dict don't lose fields.
    for key, val in preset.items():
        if key not in result:
            result[key] = val

    return result


def _readable_libs(with_set):
    """Sorted list of cuda_libs names that are '1' in with_set."""
    items = CUDA_LIBS.items()
    return sorted(lib for lib, key in items if with_set.get(key) == '1')


def _short_form(with_set):
    """
    Deterministic, stable, short fallback suffix for when the readable form
    is too long to fit the caller's max_len. Same input -> same output
    across calls and across process runs (no randomness, no dict-order
    dependence: the lib list is sorted before hashing).
    """
    libs = _readable_libs(with_set)
    digest = hashlib.sha1('-'.join(libs).encode('utf-8')).hexdigest()[:10]
    return f"{_STAGE_PREFIX}{digest}"


def stage_suffix(with_set, preset, variant, max_len=None):
    """
    Compute the Docker stage-tag suffix for a computed WITH_* set.

    Args:
        with_set: the WITH_* dict to describe (typically the output of
            compute_set()).
        preset: the preset WITH_* dict with_set was narrowed from.
        variant: the cudastack variant name (e.g. 'standard', 'minimal').
        max_len: optional maximum length for the returned suffix (e.g. 128
            minus however much of the Docker tag the caller has already
            spent on name/version/etc). Docker tags are capped at 128
            chars total, so this lets a caller enforce its own budget.
            When omitted, no limit is enforced beyond the natural default
            forms below.

    Returns:
        - When with_set == preset (byte-for-byte, e.g. because chain_above
          was empty or every package in it was undeclared): "cudastack_
          <variant>" — this reproduces today's pre-existing stage-name
          form exactly (container.py builds container_name as
          f"{name}-{package.replace(':','_')}", and for the default
          package 'cudastack:standard' that is "...-cudastack_standard"),
          so existing tags for unaffected chains don't change.
        - Otherwise: "cudastack_" + the sorted, hyphen-joined cuda_libs
          names that are on in with_set (e.g. "cudastack_cudnn-nccl").
          This is deterministic: the same with_set always produces the
          same suffix, independent of chain package order (compute_set
          already collapses order into a set).
        - If the computed suffix is longer than max_len, a short, stable
          fallback form is used instead ("cudastack_<10-hex-digest>",
          derived from the same sorted lib list so it is still
          deterministic), and if even that overflows max_len it is
          truncated. The returned suffix never exceeds max_len.
    """
    if with_set == dict(preset):
        suffix = f"{_STAGE_PREFIX}{variant}"
    else:
        libs = _readable_libs(with_set)
        joined = '-'.join(libs) if libs else 'none'
        suffix = f"{_STAGE_PREFIX}{joined}"

    if max_len is not None and len(suffix) > max_len:
        suffix = _short_form(with_set)
        if len(suffix) > max_len:
            suffix = suffix[:max_len]

    return suffix
