#!/usr/bin/env bash
set -ex

apt-get update
apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev
rm -rf /var/lib/apt/lists/*
apt-get clean



uv pip install \
        typing-extensions \
        uvicorn \
        anyio \
        starlette \
        sse-starlette \
        starlette-context \
        fastapi \
        pydantic-settings

mkdir -p /root/.cache
ln -s /data/models/stable-diffusion.cpp /root/.cache/stable-diffusion.cpp

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of stable-diffusion.cpp ${STABLE_DIFFUSION_CPP_VERSION}"
	exit 1
fi
if uv pip install --only-binary=:all: "stable-diffusion-cpp-python==${STABLE_DIFFUSION_CPP_VERSION_PY}"; then
	if [ -n "${STABLE_DIFFUSION_CPP_VERSION}" ]; then
		tarpack install "stable-diffusion-cpp-${STABLE_DIFFUSION_CPP_VERSION}" || true
	fi
	uv pip show stable-diffusion-cpp-python || true
	echo "installed" > "$TMP/.stable_diffusion_cpp"
	exit 0
fi
exit 1
