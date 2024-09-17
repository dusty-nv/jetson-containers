#---
# name: crossformer
# group: robots
# docs: docs.md
# depends: [transformers, opencv:4.10.0, jax, h5py]
# requires: '>=36'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip3 install --no-cache-dir --verbose \
        openexr \
        toolz \
        msgpack \
        nest_asyncio \
        tensorstore \
        humanize \
    && \
    pip3 install --index-url https://pypi.org/simple --no-cache-dir --verbose \
        tensorflow \
        tensorflow_hub \
        tensorflow_datasets \
        tensorflow_graphics \
    && \
    pip3 install --no-cache-dir --verbose --no-deps \
        chex \
        optax \
        distrax \
        flax \
        orbax \
        'orbax-checkpoint>=0.6.3'
        
RUN git clone --branch=main --depth=1 https://github.com/rail-berkeley/crossformer /opt/crossformer && \
    cd /opt/crossformer && \
    sed -i 's|==|>=|g' requirements.txt && \
    sed -i 's|^jax.*||g' requirements.txt && \
    sed -i 's|^chex.*||g' requirements.txt && \
    sed -i 's|^optax.*||g' requirements.txt && \
    sed -i 's|^distrax.*||g' requirements.txt && \
    sed -i 's|^flax.*||g' requirements.txt && \
    sed -i 's|^tensorflow_text.*||g' requirements.txt && \
    sed -i 's|^tensorflow_datasets.*||g' requirements.txt && \
    sed -i 's|^tensorflow_graphics.*||g' requirements.txt && \
    cat requirements.txt && \
    pip3 install --no-cache-dir --verbose -r requirements.txt && \
    pip3 install --no-cache-dir --verbose mediapy && \
    pip3 install -e .

RUN pip3 uninstall --verbose -y tensorflow_cpu_aws && \
    pip3 uninstall --verbose -y tensorflow && \
    pip3 install --index-url https://pypi.org/simple --force-reinstall --no-cache-dir --verbose 'tensorflow==2.15.0'
    
CMD ["/bin/bash"]