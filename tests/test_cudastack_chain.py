"""
Tests for jetson_containers/cudastack_chain.py — a pure, standalone module that
computes the WITH_* cudastack build-arg set for a package chain, and the
Docker stage-tag suffix for that set.

No docker, no network, no imports from packages.py/container.py: the module
under test takes only plain dicts/lists so these tests can run with plain
`python3 -m pytest tests/test_cudastack_chain.py` (or under any test runner).
"""

import importlib.util
from pathlib import Path

# Load jetson_containers/cudastack_chain.py directly by file path, WITHOUT
# importing the jetson_containers package (`import jetson_containers...`
# would run jetson_containers/__init__.py, which calls
# utils.check_dependencies() and can try to pip-install packages / hit the
# network). cudastack_chain.py is stdlib-only and self-contained, so it can
# be loaded standalone like this.
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / 'jetson_containers'
    / 'cudastack_chain.py'
)
_spec = importlib.util.spec_from_file_location('cudastack_chain', _MODULE_PATH)
cudastack_chain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cudastack_chain)

CUDA_LIBS = cudastack_chain.CUDA_LIBS
compute_set = cudastack_chain.compute_set
stage_suffix = cudastack_chain.stage_suffix


# ---------------------------------------------------------------------------
# Fixture-like helpers (plain functions, no pytest fixture magic required)
# ---------------------------------------------------------------------------


def standard_preset():
    """The 'standard' cudastack preset: everything on except gdrcopy, and
    nvpl only on SBSA (kept off here to mirror a Tegra/x86 standard build)."""
    return {
        'WITH_CUDNN': '1',
        'WITH_TENSORRT': '1',
        'WITH_NCCL': '1',
        'WITH_CUDSS': '1',
        'WITH_CUSPARSELT': '1',
        'WITH_CUTENSOR': '1',
        'WITH_GDRCOPY': '0',
        'WITH_NVPL': '0',
        'WITH_NVSHMEM': '1',
    }


def minimal_preset():
    """The 'minimal' cudastack preset: cuDNN only."""
    return {
        'WITH_CUDNN': '1',
        'WITH_TENSORRT': '0',
        'WITH_NCCL': '0',
        'WITH_CUDSS': '0',
        'WITH_CUSPARSELT': '0',
        'WITH_CUTENSOR': '0',
        'WITH_GDRCOPY': '0',
        'WITH_NVPL': '0',
        'WITH_NVSHMEM': '0',
    }


def pkg(name, cuda_libs=None):
    """Build a minimal package dict. Omitting cuda_libs entirely makes the
    package 'undeclared' (distinct from cuda_libs=[])."""
    d = {'name': name}
    if cuda_libs is not None:
        d['cuda_libs'] = cuda_libs
    return d


# ---------------------------------------------------------------------------
# CUDA_LIBS vocabulary
# ---------------------------------------------------------------------------


def test_cuda_libs_vocabulary_is_complete_and_1to1():
    expected_names = {
        'cudnn',
        'tensorrt',
        'nccl',
        'cudss',
        'cusparselt',
        'cutensor',
        'nvshmem',
        'nvpl',
        'gdrcopy',
    }
    assert set(CUDA_LIBS.keys()) == expected_names

    expected_with_keys = {
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
    assert CUDA_LIBS == expected_with_keys
    # 1:1 mapping -> no duplicate WITH_* targets
    assert len(set(CUDA_LIBS.values())) == len(CUDA_LIBS)


# ---------------------------------------------------------------------------
# compute_set
# ---------------------------------------------------------------------------


def test_compute_set_empty_chain_returns_preset():
    preset = standard_preset()
    result = compute_set([], preset)
    assert result == preset
    # must be a copy, not the same object, so callers can't mutate the preset
    assert result is not preset


def test_compute_set_jax_only():
    # jax declares cudnn + nccl only
    chain_above = [pkg('jax', cuda_libs=['cudnn', 'nccl'])]
    preset = standard_preset()
    result = compute_set(chain_above, preset)

    assert result['WITH_CUDNN'] == '1'
    assert result['WITH_NCCL'] == '1'
    # nothing else declared -> stays off even though preset allows it
    assert result['WITH_TENSORRT'] == '0'
    assert result['WITH_CUDSS'] == '0'
    assert result['WITH_CUSPARSELT'] == '0'
    assert result['WITH_CUTENSOR'] == '0'
    assert result['WITH_NVSHMEM'] == '0'
    assert result['WITH_GDRCOPY'] == '0'
    assert result['WITH_NVPL'] == '0'


def test_compute_set_jax_pytorch_opencv_crossformer_is_union():
    # crossformer-style chain: jax + pytorch + opencv, opencv declares no
    # cuda libs at all (still "declared", i.e. cuda_libs=[])
    chain_above = [
        pkg('jax', cuda_libs=['cudnn', 'nccl']),
        pkg('pytorch', cuda_libs=['cudnn', 'tensorrt', 'nccl', 'cusparselt']),
        pkg('opencv', cuda_libs=[]),
    ]
    preset = standard_preset()
    result = compute_set(chain_above, preset)

    assert result['WITH_CUDNN'] == '1'
    assert result['WITH_NCCL'] == '1'
    assert result['WITH_TENSORRT'] == '1'
    assert result['WITH_CUSPARSELT'] == '1'
    # never declared by anyone in the chain -> stays off
    assert result['WITH_CUDSS'] == '0'
    assert result['WITH_CUTENSOR'] == '0'
    assert result['WITH_NVSHMEM'] == '0'


def test_compute_set_undeclared_only_returns_full_preset():
    chain_above = [pkg('some_legacy_package')]  # no cuda_libs key at all
    preset = standard_preset()
    result = compute_set(chain_above, preset)
    assert result == preset


def test_compute_set_undeclared_package_dominates_declared_siblings():
    # one declared, narrow package plus one undeclared package -> undeclared
    # wins and the full preset passes through
    chain_above = [
        pkg('jax', cuda_libs=['cudnn']),
        pkg('mystery_package'),  # undeclared
    ]
    preset = standard_preset()
    result = compute_set(chain_above, preset)
    assert result == preset


def test_compute_set_narrow_only_declared_lib_off_in_preset_stays_off():
    # gdrcopy is declared but preset has WITH_GDRCOPY hard '0' -> stays '0'
    chain_above = [pkg('some_pkg', cuda_libs=['cudnn', 'gdrcopy'])]
    preset = standard_preset()
    result = compute_set(chain_above, preset)
    assert result['WITH_GDRCOPY'] == '0'
    # result must always be a subset of what the preset allows
    for key, val in result.items():
        if val == '1':
            assert preset[key] == '1'


def test_compute_set_result_is_always_subset_of_preset_minimal():
    # against the minimal preset, even declaring everything can't turn more on
    chain_above = [pkg('kitchen_sink', cuda_libs=list(CUDA_LIBS.keys()))]
    preset = minimal_preset()
    result = compute_set(chain_above, preset)
    for key, val in preset.items():
        if val == '0':
            assert result[key] == '0'
    assert result['WITH_CUDNN'] == '1'  # cudnn always on, and preset allows it


def test_compute_set_trt_consumer_chain_pulls_in_tensorrt():
    # e.g. a tensorrt_edge_llm style consumer package
    chain_above = [pkg('tensorrt_edge_llm', cuda_libs=['cudnn', 'tensorrt'])]
    preset = standard_preset()
    result = compute_set(chain_above, preset)
    assert result['WITH_TENSORRT'] == '1'
    assert result['WITH_NCCL'] == '0'


def test_compute_set_trt_consumer_chain_narrowed_by_minimal_preset():
    # same declared chain, but against the minimal preset TRT stays off
    chain_above = [pkg('tensorrt_edge_llm', cuda_libs=['cudnn', 'tensorrt'])]
    preset = minimal_preset()
    result = compute_set(chain_above, preset)
    assert result['WITH_TENSORRT'] == '0'


def test_compute_set_cudnn_always_on_even_if_never_declared():
    # background: config.py:160 hard-codes WITH_CUDNN='1' in every preset;
    # compute_set must keep reflecting that (never turn cudnn off on its own)
    chain_above = [pkg('nccl_only_pkg', cuda_libs=['nccl'])]
    preset = standard_preset()
    result = compute_set(chain_above, preset)
    assert result['WITH_CUDNN'] == '1'


def test_compute_set_does_not_mutate_inputs():
    chain_above = [pkg('jax', cuda_libs=['cudnn', 'nccl'])]
    chain_copy = [dict(p) for p in chain_above]
    preset = standard_preset()
    preset_copy = dict(preset)
    compute_set(chain_above, preset)
    assert chain_above == chain_copy
    assert preset == preset_copy


# ---------------------------------------------------------------------------
# stage_suffix
# ---------------------------------------------------------------------------


def test_stage_suffix_matches_todays_tag_when_set_equals_preset():
    # Today (pre-cudastack-chain), container.py builds the stage name as
    #   f"{name}-{package.replace(':','_')}"
    # for 'cudastack:standard' that yields "...-cudastack_standard"
    # tag. An undeclared chain resolves to the full preset via compute_set,
    # so stage_suffix must reproduce "cudastack_standard" byte-for-byte in
    # that case, to keep existing tags stable.
    preset = standard_preset()
    effective = compute_set([pkg('undeclared_pkg')], preset)  # == preset
    suffix = stage_suffix(effective, preset, 'standard')
    assert suffix == 'cudastack_standard'


def test_stage_suffix_equal_sets_are_variant_name_form_regardless_of_chain():
    preset = standard_preset()
    # directly passing the preset itself (empty-chain case) must match too
    effective = compute_set([], preset)
    assert stage_suffix(effective, preset, 'standard') == 'cudastack_standard'


def test_stage_suffix_narrowed_set_lists_sorted_lib_names():
    preset = standard_preset()
    chain_above = [pkg('jax', cuda_libs=['cudnn', 'nccl'])]
    effective = compute_set(chain_above, preset)
    suffix = stage_suffix(effective, preset, 'standard')
    assert suffix == 'cudastack_cudnn-nccl'


def test_stage_suffix_is_deterministic_across_calls():
    preset = standard_preset()
    chain_above = [
        pkg('pytorch', cuda_libs=['cudnn', 'tensorrt', 'nccl', 'cusparselt']),
        pkg('jax', cuda_libs=['cudnn', 'nccl']),
    ]
    effective = compute_set(chain_above, preset)
    s1 = stage_suffix(effective, preset, 'standard')
    s2 = stage_suffix(effective, preset, 'standard')
    assert s1 == s2
    # order of packages in chain_above must not affect the suffix
    reordered = list(reversed(chain_above))
    effective2 = compute_set(reordered, preset)
    assert stage_suffix(effective2, preset, 'standard') == s1


def test_stage_suffix_trt_consumer_chain():
    preset = standard_preset()
    chain_above = [pkg('tensorrt_edge_llm', cuda_libs=['cudnn', 'tensorrt'])]
    effective = compute_set(chain_above, preset)
    suffix = stage_suffix(effective, preset, 'standard')
    assert suffix == 'cudastack_cudnn-tensorrt'


def test_stage_suffix_enforces_max_length():
    preset = standard_preset()
    # declare every lib so the readable form would be long
    chain_above = [pkg('kitchen_sink', cuda_libs=list(CUDA_LIBS.keys()))]
    effective = compute_set(chain_above, preset)
    readable = stage_suffix(effective, preset, 'standard')
    assert len(readable) > 10

    short = stage_suffix(effective, preset, 'standard', max_len=10)
    assert len(short) <= 10
    # must still be deterministic
    assert stage_suffix(effective, preset, 'standard', max_len=10) == short


def test_stage_suffix_max_length_never_exceeded_even_for_tiny_limits():
    preset = standard_preset()
    chain_above = [pkg('kitchen_sink', cuda_libs=list(CUDA_LIBS.keys()))]
    effective = compute_set(chain_above, preset)
    for limit in (1, 3, 5, 8, 20, 128):
        suffix = stage_suffix(effective, preset, 'standard', max_len=limit)
        assert len(suffix) <= limit


def test_stage_suffix_fallback_is_stable_short_form_not_random():
    preset = standard_preset()
    chain_above = [pkg('kitchen_sink', cuda_libs=list(CUDA_LIBS.keys()))]
    effective = compute_set(chain_above, preset)
    short_a = stage_suffix(effective, preset, 'standard', max_len=12)
    short_b = stage_suffix(effective, preset, 'standard', max_len=12)
    assert short_a == short_b


def test_stage_suffix_under_limit_uses_readable_form():
    preset = standard_preset()
    chain_above = [pkg('jax', cuda_libs=['cudnn', 'nccl'])]
    effective = compute_set(chain_above, preset)
    suffix = stage_suffix(effective, preset, 'standard', max_len=128)
    assert suffix == 'cudastack_cudnn-nccl'


if __name__ == '__main__':
    import pytest

    raise SystemExit(pytest.main([__file__, '-v']))
