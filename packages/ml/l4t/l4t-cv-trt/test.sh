#!/usr/bin/env bash
# l4t-cv-trt smoke test (6 checks + version manifest + obligation probes).
# Runs via `jetson-containers build --test-only` (mounted at /test), or directly:
#   docker run --runtime nvidia --rm -v $PWD:/test IMAGE bash /test/test.sh
# Set EXPECT_TRT (e.g. -e EXPECT_TRT=10.7) to also assert the TensorRT major.minor version.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPECT_TRT="${EXPECT_TRT:-}"
export WORK="$(mktemp -d /tmp/l4t-cv-trt.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

PY="$(command -v python3)"
FAIL=0

echo "=== [6/6 first] python + torch + torchvision + cv2 + tensorrt checks ==="
"$PY" "$HERE/test_smoke.py" || FAIL=1

echo "=== ffmpeg encode+decode ==="
FF_OK=0
if command -v ffmpeg >/dev/null 2>&1; then
    ENC=libx264
    ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 || ENC=mpeg4
    if ffmpeg -hide_banner -loglevel error -y -f lavfi -i testsrc=duration=1:size=320x240:rate=25 \
            -c:v "$ENC" -pix_fmt yuv420p "$WORK/out.mp4" &&
       NF=$(ffmpeg -hide_banner -i "$WORK/out.mp4" -map 0:v:0 -f null - 2>&1 |
            grep -o 'frame= *[0-9]*' | tail -1 | tr -dc 0-9) &&
       [ "${NF:-0}" -ge 24 ]; then
        FF_OK=1
        echo "PASS ffmpeg: encoder=$ENC decoded_frames=$NF"
    else
        echo "FAIL ffmpeg: encode/decode (encoder=$ENC frames=${NF:-none})"
    fi
else
    echo "FAIL ffmpeg: not on PATH"
fi
[ "$FF_OK" = 1 ] || FAIL=1

echo "=== manifest ==="
"$PY" - <<'PYEOF'
import json, re, subprocess, sys
m = {"python": sys.version.split()[0]}
def tryv(k, f):
    try: m[k] = f()
    except Exception as e: m[k] = "ERR:%s" % type(e).__name__
import torch
tryv("torch", lambda: torch.__version__)
tryv("cuda", lambda: torch.version.cuda)
tryv("cudnn", lambda: str(torch.backends.cudnn.version()))
def tv():
    import torchvision; return torchvision.__version__
tryv("torchvision", tv)
def cv():
    import cv2; return cv2.__version__
tryv("cv2", cv)
def trt():
    import tensorrt; return tensorrt.__version__
tryv("tensorrt", trt)
def ff():
    o = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
    return re.search(r"ffmpeg version (\S+)", o).group(1)
tryv("ffmpeg", ff)
print("MANIFEST " + json.dumps(m))
PYEOF

echo "=== obligation probes (informational) ==="
"$PY" - <<'PYEOF'
import re
try:
    import torch
    print("PROBE torch.backends.cudnn.version() =", torch.backends.cudnn.version())
except Exception as e:
    print("PROBE cudnn: ERR", e)
try:
    import cv2
    info = cv2.getBuildInformation()
    for l in info.splitlines():
        if re.search(r"NVIDIA GPU arch|NVIDIA PTX archs|FFMPEG|avcodec|CUDA:|cuDNN:", l):
            print("PROBE cv2 build-info:", l.strip())
    print("PROBE cv2 file:", cv2.__file__)
except Exception as e:
    print("PROBE cv2 build-info: ERR", e)
PYEOF
CV2SO="$("$PY" -c 'import cv2,glob,os;d=os.path.dirname(cv2.__file__);print((glob.glob(d+"/cv2*.so")+glob.glob(d+"/python*/cv2*.so")+[""])[0])' 2>/dev/null)"
if [ -n "$CV2SO" ]; then
    echo "PROBE ldd cv2 ($CV2SO) libav*:"
    ldd "$CV2SO" 2>&1 | grep -E 'libav(codec|format|util)|libsw|not found' || echo "  (no libav* linked, or none found)"
else
    echo "PROBE ldd cv2: cv2 .so not located"
fi
for lib in libnvinfer.so.10 libcudnn.so.9; do
    echo "PROBE resolved $lib: $(ldconfig -p 2>/dev/null | awk -v l="$lib" '$1==l{print $NF; exit}') -> $(readlink -f "$(ldconfig -p 2>/dev/null | awk -v l="$lib" '$1==l{print $NF; exit}')" 2>/dev/null)"
done
echo "PROBE pip-guard: uv pip install --dry-run opencv-python-headless"
if command -v uv >/dev/null 2>&1; then
    OUT="$(uv pip install --dry-run opencv-python-headless 2>&1)"
    echo "$OUT" | sed 's/^/  /' | head -15
    if echo "$OUT" | grep -qiE 'Would install.*opencv|\+ opencv-python-headless'; then
        echo "PROBE pip-guard RESULT: NOT guarded (cv2 would be replaced)"
    else
        echo "PROBE pip-guard RESULT: guarded or no-op (cv2 not reinstalled)"
    fi
else
    echo "PROBE pip-guard: uv not found"
fi
env | grep -E '^(PIP_CONSTRAINT|UV_CONSTRAINT|PIP_INDEX_URL|UV_INDEX|UV_DEFAULT_INDEX|UV_EXTRA_INDEX_URL)=' | sed 's/^/  env: /'

echo "=== result ==="
if [ "$FAIL" = 0 ]; then echo "l4t-cv-trt smoke: ALL PASS"; else echo "l4t-cv-trt smoke: FAILED"; fi
exit "$FAIL"
