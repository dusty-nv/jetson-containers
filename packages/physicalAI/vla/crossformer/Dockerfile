#---
# name: crossformer
# group: vla
# docs: docs.md
# depends: [transformers, opencv, jax, h5py]
# requires: '>=36'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN uv pip install \
        openexr \
        toolz \
        msgpack \
        nest_asyncio \
        tensorstore \
        humanize \
    && \
    uv pip install --index-url https://pypi.org/simple \
        tensorflow \
        tensorflow_hub \
        tensorflow_datasets \
        tensorflow_graphics \
    && \
    uv pip install --no-deps \
        chex \
        optax \
        distrax \
        flax \
        orbax \
        'orbax-checkpoint>=0.6.3'

RUN git clone --branch=main --depth=1 https://github.com/dusty-nv/crossformer /opt/crossformer && \
    cd /opt/crossformer && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i 's/~=/>=/g' requirements.txt && \
    sed -i 's|^jax.*||g' requirements.txt && \
    sed -i 's|^chex.*||g' requirements.txt && \
    sed -i 's|^optax.*||g' requirements.txt && \
    sed -i 's|^distrax.*||g' requirements.txt && \
    sed -i 's|^flax.*||g' requirements.txt && \
    sed -i 's|^tensorflow_text.*||g' requirements.txt && \
    sed -i 's|^tensorflow_datasets.*||g' requirements.txt && \
    sed -i 's|^tensorflow_graphics.*||g' requirements.txt && \
    cat requirements.txt && \
    uv pip install -r requirements.txt && \
    uv pip install mediapy && \
    uv pip install -e .

RUN uv pip uninstall --verbose -y tensorflow_cpu_aws && \
    uv pip uninstall --verbose -y tensorflow && \
    uv pip install tensorflow tensorflow_probability

COPY benchmark.py /opt/crossformer

CMD ["/bin/bash"]
