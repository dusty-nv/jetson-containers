# l4t-cv-trt

JetPack 6.x computer-vision + TensorRT stack: `python`, `ffmpeg`, `opencv:4.14.0`, `pytorch`,
`torchvision` and `cudastack:standard` (cuDNN, TensorRT, NCCL, ...). Requires L4T >= 36.

## Pinning versions

- **OpenCV** is pinned by the versioned dependency `opencv:4.14.0` in `l4t-cv-trt.json`
  (falls back to `4.12.0` if 4.14.0 does not build: edit the dependency).
- **TensorRT** is selected by `cudastack` at resolve time, not by this package. Put
  `TENSORRT_VERSION=10.7` in the build environment (needs the `TENSORRT_VERSION` override in
  `packages/cuda/cudastack/config.py`):

```bash
CUDA_ARCH=tegra-aarch64 L4T_VERSION=36.4.3 JETPACK_VERSION=6.2 CUDA_VERSION=12.6 \
TENSORRT_VERSION=10.7 jetson-containers build l4t-cv-trt
```

The smoke test asserts `tensorrt` is `10.7.x` (override with `EXPECT_TRT=...`), so a build that
resolved a different TensorRT fails loudly.

## Smoke test (`test.sh`)

Six checks: python 3.10; torch CUDA matmul; `torchvision.ops.nms` on CUDA; `cv2.cuda` device
count >= 1; TensorRT builds an engine from a tiny in-test ONNX (64 MiB workspace); ffmpeg 1 s
`testsrc` encode + decode. Exits non-zero on any failure. It also prints a `MANIFEST {...}` JSON
line and the obligation probes (cuDNN version, cv2 build info, `ldd` of cv2 for libav*, resolved
`libnvinfer.so.10` / `libcudnn.so.9`, pip-guard dry-run; the probes are informational).

```bash
jetson-containers build --test-only l4t-cv-trt l4t-cv-trt
# or, on an already built image:
docker run --runtime nvidia --rm -v $PWD/packages/ml/l4t/l4t-cv-trt:/test IMAGE bash /test/test.sh
```
