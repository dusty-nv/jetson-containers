#---
# name: openpi
# group: robots
# docs: docs.md
# depends: [jax, pytorch, transformers, lerobot, opencv, pyav]
# requires: '>=36'
# test: [test.sh, test.py]
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

WORKDIR /opt/openpi

# Clone and build openpi
RUN git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi /opt/openpi && \
    cd /opt/openpi && \
    sed -i '/"torch[^"]*",/d' pyproject.toml && \
    sed -i '/"opencv-python[^"]*",/d' pyproject.toml && \
    sed -i '/"transformers[^"]*",/d' pyproject.toml && \
    sed -i '/"lerobot[^"]*",/d' pyproject.toml && \
    sed -i '/"jax[^"]*",/d' pyproject.toml && \
    sed -i '/"av[^"]*",/d' pyproject.toml && \
    sed -i '/"flax[^"]*",/d'  pyproject.toml && \
    sed -i '/"jaxtyping[^"]*",/d'  pyproject.toml && \
    sed -i '/"orbax-checkpoint[^"]*",/d'  pyproject.toml && \
    sed -i '/"equinox[^"]*",/d'  pyproject.toml && \
    sed -i '/"gym-aloha[^"]*",/d' pyproject.toml && \
    cat -n pyproject.toml

# Ensure CUDA and cuDNN are installed first
# RUN apt-get update && apt-get install -y \
#     cuda-toolkit-12-8 \
#     libcudnn9-dev

# RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install mujoco system packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-dev \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first
RUN uv pip install --no-cache-dir --extra-index-url https://pypi.org/simple --index-url https://pypi.org/simple 'av>=12.0.5,<13.0.0' && \
    uv pip install --no-deps flax==0.12.0 && \
    uv pip install --no-deps equinox>=0.11.8 && \
    uv pip install --no-deps orbax-checkpoint==0.11.16 && \
    uv pip install --no-deps jaxtyping==0.3.2

# # Install mujoco 2.3.7 which is required by gym-aloha
# RUN uv pip install --no-cache-dir --only-binary :all: mujoco==2.3.7

# Install gym-aloha without dependencies first
RUN uv pip install --no-cache-dir --no-deps gym-aloha>=0.1.1

COPY benchmark_pi0_droid.py /opt/openpi/benchmark_pi0_droid.py

# Install local packages
RUN cd /opt/openpi && \
    uv pip install -e packages/openpi-client && \
    uv pip install -e .

RUN uv pip install jaxlib jax_cuda12_plugin jax_cuda12_pjrt opt_einsum && \
    uv pip install --no-dependencies jax

RUN uv pip install --no-deps optax importlib_resources humanize tensorstore simplejson datasets pandas pyarrow pytz multiprocess metadata dill xxhash jsonlines draccus mergedeep typing_inspect mypy_extensions chex etils toolz
RUN python3 -c "import jax; import jaxlib; print('JAX version:', jax.__version__); print('JAXLIB version:', jaxlib.__version__); print('Devices:', jax.devices())"

CMD /bin/bash
