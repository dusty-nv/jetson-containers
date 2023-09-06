#---
# name: auto_gptq
# group: llm
# config: config.py
# depends: [pytorch, bitsandbytes, transformers]
# requires: '>=34.1.0'
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCH_CUDA_ARCH_LIST

WORKDIR /opt

# 8/28/2023 - build errors in main (v0.4.2 is tag prior)
ARG AUTOGPTQ_BRANCH="v0.4.2"

#ADD https://api.github.com/repos/PanQiWei/AutoGPTQ/git/refs/heads/${AUTOGPTQ_BRANCH} /tmp/autogptq_version.json

RUN git clone --branch=${AUTOGPTQ_BRANCH} --depth=1 https://github.com/PanQiWei/AutoGPTQ.git && \
    cd AutoGPTQ && \
    sed 's|^    p = int(subprocess.*| \ \ \ p = os.cpu_count()|g' -i setup.py && \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/auto_gptq*.whl /opt

RUN pip3 install --no-cache-dir --verbose /opt/auto_gptq*.whl

# add memory stats to benchmark
RUN sed '/    logger.info(f"model device map.*/a\ \ \ \ logger.info(f"torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated()}")' -i /opt/AutoGPTQ/examples/benchmark/generation_speed.py && \
    sed '/    benchmark_generation_speed(model.*/a\ \ \ \ logger.info(f"torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated()}")' -i /opt/AutoGPTQ/examples/benchmark/generation_speed.py && \
    cat AutoGPTQ/examples/benchmark/generation_speed.py
    
WORKDIR /

# make sure it loads
RUN pip3 show auto-gptq && python3 -c 'import auto_gptq'