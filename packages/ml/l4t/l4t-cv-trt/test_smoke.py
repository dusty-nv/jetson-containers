#!/usr/bin/env python3
# l4t-cv-trt smoke checks 1-5 (ffmpeg is checked in test.sh). Exit 1 on any failure.
# Keep memory small: tiny tensors, 64 MiB TensorRT workspace, everything in $WORK or /tmp.
import os
import sys
import tempfile

EXPECT_TRT = os.environ.get("EXPECT_TRT", "")  # e.g. "10.7"; empty = accept what cudastack installed
WORK = os.environ.get("WORK") or tempfile.mkdtemp(prefix="l4t-cv-trt.")
results = []


def check(name):
    def deco(fn):
        try:
            detail = fn()
            results.append((name, True, detail))
            print(f"PASS {name}: {detail}")
        except BaseException as e:  # noqa: BLE001 - report every failure kind
            results.append((name, False, repr(e)))
            print(f"FAIL {name}: {e!r}")
        return fn
    return deco


@check("python 3.10")
def _():
    assert sys.version_info[:2] == (3, 10), sys.version
    return sys.version.split()[0]


@check("torch CUDA matmul")
def _():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    a = torch.randn(128, 128, device="cuda")
    b = torch.randn(128, 128, device="cuda")
    c = (a @ b).cpu()
    ref = a.cpu() @ b.cpu()
    assert torch.allclose(c, ref, atol=1e-2), "matmul mismatch"
    return f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}"


@check("torchvision.ops.nms CUDA")
def _():
    import torch
    import torchvision
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                         dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    keep = torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist()
    assert keep == [0, 2], keep
    return f"torchvision {torchvision.__version__} keep={keep}"


@check("cv2.cuda device count")
def _():
    import cv2
    n = cv2.cuda.getCudaEnabledDeviceCount()
    assert n >= 1, f"cv2.cuda.getCudaEnabledDeviceCount()={n} (cv2 {cv2.__version__})"
    return f"cv2 {cv2.__version__} cuda devices={n}"


@check(f"tensorrt {EXPECT_TRT + '.x ' if EXPECT_TRT else ''}engine from tiny ONNX")
def _():
    import ctypes
    import tensorrt as trt
    if EXPECT_TRT:
        assert trt.__version__.startswith(EXPECT_TRT + "."), (
            f"tensorrt {trt.__version__} != {EXPECT_TRT}.x "
            f"(build with TENSORRT_VERSION={EXPECT_TRT} in the environment)")
    # the python bindings must match the libnvinfer the loader resolves (no host/image mix)
    lib = ctypes.CDLL("libnvinfer.so." + trt.__version__.split(".")[0])
    v = lib.getInferLibVersion()  # TensorRT 10: major * 10000 + minor * 100 + patch
    assert (v // 10000, v // 100 % 100) == tuple(int(x) for x in trt.__version__.split(".")[:2]), (
        f"tensorrt python {trt.__version__} != libnvinfer {v}")
    import torch
    onnx_path = os.path.join(WORK, "tiny.onnx")
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU()).eval()
    export_kw = dict(input_names=["x"], output_names=["y"], opset_version=13)
    try:  # torch >= 2.5 may default to the dynamo exporter; keep the TorchScript one
        torch.onnx.export(model, torch.zeros(1, 1, 16, 16), onnx_path, dynamo=False, **export_kw)
    except TypeError:  # older torch has no 'dynamo' argument
        torch.onnx.export(model, torch.zeros(1, 1, 16, 16), onnx_path, **export_kw)
    del model
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the default in TRT 10
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    assert ok, [str(parser.get_error(i)) for i in range(parser.num_errors)]
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    blob = builder.build_serialized_network(network, config)
    assert blob is not None and blob.nbytes > 0, "engine build returned nothing"  # IHostMemory
    return f"tensorrt {trt.__version__} engine bytes={blob.nbytes}"


failed = [n for n, ok, _ in results if not ok]
print(f"python checks: {len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
