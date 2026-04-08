#!/usr/bin/env python3
"""
Tests for the CUDA arch replacement logic in generate_diff.py.

Run with:
    python3 packages/llm/vllm/patches/test_generate_diff.py
or:
    python3 -m pytest packages/llm/vllm/patches/test_generate_diff.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

# Stub out subprocess/urllib so importing generate_diff doesn't need torch/network
import unittest.mock as mock
with mock.patch("subprocess.run"), mock.patch("urllib.request.urlopen"):
    from generate_diff import modify_cmake_archs


# ---------------------------------------------------------------------------
# Realistic CMakeLists.txt snippet covering all intersection patterns
# ---------------------------------------------------------------------------
CMAKE_TEMPLATE = """\
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0;12.1")
else()
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

  cuda_archs_loose_intersection(CUDA_ARCHS
    "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")

  cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(MARLIN_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(MARLIN_BF16_ARCHS "8.0+PTX;9.0+PTX" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(MARLIN_FP8_ARCHS "8.9;12.0;12.1" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(MARLIN_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(ALLSPARK_ARCHS "8.0;8.6;8.7;8.9" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(MACHETE_ARCHS "9.0a" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(W4A8_ARCHS "9.0a" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(HADACORE_ARCHS "8.0+PTX;9.0+PTX" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(DSV3_FUSED_ARCHS "9.0a;10.0f;11.0f" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(BW_SM100_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(BW_SM120_ARCHS "12.0f" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(MLA_ARCHS "10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(FP4_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
  cuda_archs_loose_intersection(MARLIN_MOE_FP8_ARCHS "8.9;12.0;12.1" "${CUDA_ARCHS}")
"""

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_intersections(output: str) -> dict:
    """Parse all cuda_archs_loose_intersection lines into {VAR: first_arg}."""
    import re
    result = {}
    for line in output.splitlines():
        # single-line form only (two-line was already collapsed by modify_cmake_archs)
        m = re.match(r'\s*cuda_archs_loose_intersection\s*\(\s*(\w+)\s+"([^"]+)"', line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


def _get_supported_archs(output: str) -> str:
    """Return the value from set(CUDA_SUPPORTED_ARCHS ...)."""
    import re
    m = re.search(r'set\(CUDA_SUPPORTED_ARCHS\s+"([^"]+)"\)', output)
    return m.group(1) if m else None


def _run(target: str) -> dict:
    """Run modify_cmake_archs with the given target and return parsed intersections."""
    os.environ["TORCH_CUDA_ARCH_LIST"] = target
    output = modify_cmake_archs(CMAKE_TEMPLATE)
    del os.environ["TORCH_CUDA_ARCH_LIST"]
    return output, _get_intersections(output), _get_supported_archs(output)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCudaSupportedArchs(unittest.TestCase):
    """Step 1: CUDA_SUPPORTED_ARCHS block is always collapsed to the target."""

    def _assert_supported_archs(self, target):
        output, _, supported = _run(target)
        self.assertEqual(supported, target,
            f"CUDA_SUPPORTED_ARCHS should be '{target}', got '{supported}'")
        # The if/elseif/endif block must be gone
        self.assertNotIn("elseif", output)

    def test_target_87(self):  self._assert_supported_archs("8.7")
    def test_target_90(self):  self._assert_supported_archs("9.0")
    def test_target_103(self): self._assert_supported_archs("10.3")
    def test_target_110(self): self._assert_supported_archs("11.0")
    def test_target_120(self): self._assert_supported_archs("12.0")


class TestCmakeVarRefAlwaysReplaced(unittest.TestCase):
    """${CUDA_SUPPORTED_ARCHS} variable references are always replaced."""

    def _assert_cuda_archs_replaced(self, target):
        _, intersections, _ = _run(target)
        # The two-line CUDA_ARCHS intersection uses ${CUDA_SUPPORTED_ARCHS}
        self.assertIn("CUDA_ARCHS", intersections)
        self.assertEqual(intersections["CUDA_ARCHS"], target)

    def test_target_87(self):  self._assert_cuda_archs_replaced("8.7")
    def test_target_103(self): self._assert_cuda_archs_replaced("10.3")
    def test_target_120(self): self._assert_cuda_archs_replaced("12.0")


class TestTarget87(unittest.TestCase):
    """Target SM 8.7 (Jetson Orin) — only SM 8.x and below filters replaced."""

    def setUp(self):
        _, self.ints, _ = _run("8.7")

    # Should be replaced (same major or lower)
    def test_marlin_archs_replaced(self):
        self.assertEqual(self.ints["MARLIN_ARCHS"], "8.7")       # "8.0+PTX"

    def test_marlin_sm75_replaced(self):
        self.assertEqual(self.ints["MARLIN_SM75_ARCHS"], "8.7")  # "7.5"

    def test_marlin_bf16_replaced(self):
        self.assertEqual(self.ints["MARLIN_BF16_ARCHS"], "8.7")  # "8.0+PTX;9.0+PTX" — 8.x present

    def test_marlin_fp8_replaced(self):
        self.assertEqual(self.ints["MARLIN_FP8_ARCHS"], "8.7")   # "8.9;12.0;12.1" — 8.x present

    def test_marlin_moe_fp8_replaced(self):
        self.assertEqual(self.ints["MARLIN_MOE_FP8_ARCHS"], "8.7")  # "8.9;12.0;12.1"

    def test_marlin_other_replaced(self):
        self.assertEqual(self.ints["MARLIN_OTHER_ARCHS"], "8.7") # "7.5;8.0+PTX"

    def test_allspark_replaced(self):
        self.assertEqual(self.ints["ALLSPARK_ARCHS"], "8.7")     # "8.0;8.6;8.7;8.9"

    def test_hadacore_replaced(self):
        self.assertEqual(self.ints["HADACORE_ARCHS"], "8.7")     # "8.0+PTX;9.0+PTX" — 8.x present

    def test_scaled_mm_2x_replaced(self):
        self.assertEqual(self.ints["SCALED_MM_2X_ARCHS"], "8.7") # "7.5;8.0;8.7;8.9+PTX"

    # Should NOT be replaced (all archs have major > 8)
    def test_scaled_mm_hopper_unchanged(self):
        self.assertEqual(self.ints["SCALED_MM_ARCHS"], "9.0a;")  # "9.0a"

    def test_machete_unchanged(self):
        self.assertEqual(self.ints["MACHETE_ARCHS"], "9.0a")     # "9.0a"

    def test_w4a8_unchanged(self):
        self.assertEqual(self.ints["W4A8_ARCHS"], "9.0a")        # "9.0a"

    def test_dsv3_fused_unchanged(self):
        self.assertEqual(self.ints["DSV3_FUSED_ARCHS"], "9.0a;10.0f;11.0f")

    def test_bw_sm100_unchanged(self):
        self.assertEqual(self.ints["BW_SM100_ARCHS"], "10.0f;11.0f")

    def test_bw_sm120_unchanged(self):
        self.assertEqual(self.ints["BW_SM120_ARCHS"], "12.0f")

    def test_mla_unchanged(self):
        self.assertEqual(self.ints["MLA_ARCHS"], "10.0a;10.1a;10.3a;12.0a;12.1a")

    def test_fp4_unchanged(self):
        self.assertEqual(self.ints["FP4_ARCHS"], "12.0a;12.1a")


class TestTarget103(unittest.TestCase):
    """Target SM 10.3 — SM 10.x and below filters replaced, SM 11+ unchanged."""

    def setUp(self):
        _, self.ints, _ = _run("10.3")

    def test_marlin_replaced(self):
        self.assertEqual(self.ints["MARLIN_ARCHS"], "10.3")       # 8 <= 10

    def test_machete_replaced(self):
        self.assertEqual(self.ints["MACHETE_ARCHS"], "10.3")      # 9 <= 10

    def test_dsv3_fused_replaced(self):
        self.assertEqual(self.ints["DSV3_FUSED_ARCHS"], "10.3")   # 9 <= 10

    def test_bw_sm100_replaced(self):
        self.assertEqual(self.ints["BW_SM100_ARCHS"], "10.3")     # 10 <= 10

    def test_mla_replaced(self):
        self.assertEqual(self.ints["MLA_ARCHS"], "10.3")          # 10 <= 10

    def test_bw_sm120_unchanged(self):
        self.assertEqual(self.ints["BW_SM120_ARCHS"], "12.0f")    # 12 > 10

    def test_fp4_unchanged(self):
        self.assertEqual(self.ints["FP4_ARCHS"], "12.0a;12.1a")   # 12 > 10


class TestTarget110(unittest.TestCase):
    """Target SM 11.0 (Jetson Thor) — SM 11.x and below replaced, SM 12+ unchanged."""

    def setUp(self):
        _, self.ints, _ = _run("11.0")

    def test_machete_replaced(self):
        self.assertEqual(self.ints["MACHETE_ARCHS"], "11.0")      # 9 <= 11

    def test_bw_sm100_replaced(self):
        self.assertEqual(self.ints["BW_SM100_ARCHS"], "11.0")     # 10,11 <= 11

    def test_mla_replaced(self):
        self.assertEqual(self.ints["MLA_ARCHS"], "11.0")          # 10 <= 11

    def test_bw_sm120_unchanged(self):
        self.assertEqual(self.ints["BW_SM120_ARCHS"], "12.0f")    # 12 > 11

    def test_fp4_unchanged(self):
        self.assertEqual(self.ints["FP4_ARCHS"], "12.0a;12.1a")   # 12 > 11


class TestTarget120(unittest.TestCase):
    """Target SM 12.0 — everything replaced (nothing above 12 in template)."""

    def setUp(self):
        _, self.ints, _ = _run("12.0")

    def test_machete_replaced(self):
        self.assertEqual(self.ints["MACHETE_ARCHS"], "12.0")

    def test_bw_sm100_replaced(self):
        self.assertEqual(self.ints["BW_SM100_ARCHS"], "12.0")

    def test_bw_sm120_replaced(self):
        self.assertEqual(self.ints["BW_SM120_ARCHS"], "12.0")

    def test_fp4_replaced(self):
        self.assertEqual(self.ints["FP4_ARCHS"], "12.0")

    def test_mla_replaced(self):
        self.assertEqual(self.ints["MLA_ARCHS"], "12.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
