#---
# name: nemo
# group: ml
# depends: [transformers, torchaudio, numba]
# test: [test_qa.py]
# docs: NVIDIA NeMo for ASR/NLP/TTS https://nvidia.github.io/NeMo/
# notes: https://research.nvidia.com/labs/conv-ai/
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


# Nemo needs more recent OpenFST than focal apt has
ARG FST_VERSION=1.8.2
RUN cd /tmp && \
    wget $WGET_FLAGS https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-${FST_VERSION}.tar.gz && \
    tar -xzvf openfst-${FST_VERSION}.tar.gz && \
    cd openfst-${FST_VERSION} && \
    ./configure --enable-grm && \
    make -j$(nproc) && \
    make install && \
    cd ../ && \
    rm -rf openfst

# install nemo_toolkit
RUN uv pip install nemo_toolkit['all']

# libopencc.so.1 needed by: nemo/collections/common/tokenizers/chinese_tokenizers.py
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          libopencc-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# patch: cannot import name 'GradBucket' from 'torch.distributed'
RUN NEMO_PATH="$(uv pip show nemo_toolkit | grep Location: | cut -d' ' -f2)/nemo" && \
    sed -i '/from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook/d' $NEMO_PATH/collections/nlp/parts/nlp_overrides.py

# patch: Unexpected key(s) in state_dict: "bert_model.embeddings.position_ids".
# with:  nemo_toolkit 1.19.1, transformers 4.31.0
#RUN uv pip install 'transformers<4.31'

# make sure it loads
RUN uv pip show nemo_toolkit && python3 -c 'import nemo; print(nemo.__version__)'

# clone the repo for the examples
RUN git clone --depth=1 https://github.com/NVIDIA/NeMo /opt/nemo

# install apex (needed for megatron)
ARG TORCH_CUDA_ARCH_LIST

RUN cd /opt && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    sed 's|if "all_gather_into_tensor"|if torch.distributed.is_available() and "all_gather_into_tensor"|g' -i apex/transformer/utils.py && \
    sed 's|if "all_gather_into_tensor"|if torch.distributed.is_available() and "all_gather_into_tensor"|g' -i apex/transformer/tensor_parallel/layers.py && \
    sed 's|if "reduce_scatter_tensor"|if torch.distributed.is_available() and "reduce_scatter_tensor"|g' -i apex/transformer/tensor_parallel/layers.py && \
    sed 's|if "all_gather_into_tensor"|if torch.distributed.is_available() and "all_gather_into_tensor"|g' -i apex/transformer/tensor_parallel/mappings.py && \
    sed 's|if "reduce_scatter_tensor"|if torch.distributed.is_available() and "reduce_scatter_tensor"|g' -i apex/transformer/tensor_parallel/mappings.py && \
    head -n15 apex/transformer/utils.py && \
    head -n75 apex/transformer/tensor_parallel/layers.py && \
    head -n40 apex/transformer/tensor_parallel/mappings.py && \
    uv pip install -r requirements.txt && \
    python3 setup.py --verbose bdist_wheel --cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam && \
    cp dist/apex*.whl /opt && \
    rm -rf /opt/apex

RUN uv pip install /opt/apex*.whl

# module 'torch.distributed' has no attribute 'is_initialized'
RUN PYTHON_ROOT=`uv pip show nemo_toolkit | grep Location: | cut -d' ' -f2` && \
    sed 's|^        master_device =.*|        master_device = True|g' -i $PYTHON_ROOT/nemo/collections/nlp/modules/common/megatron/megatron_utils.py && \
    sed 's|not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0|torch.distributed.is_available() and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)|g' -i $PYTHON_ROOT/nemo/collections/nlp/modules/common/megatron/megatron_utils.py && \
    sed 's|if torch.distributed.is_initialized()|if torch.distributed.is_available() and torch.distributed.is_initialized()|g' -i $PYTHON_ROOT/nemo/collections/nlp/modules/common/megatron/megatron_utils.py && \
    sed 's|self.trainer.strategy.setup_environment()|#self.trainer.strategy.setup_environment()|g' -i $PYTHON_ROOT/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py && \
    head -225 $PYTHON_ROOT/nemo/collections/nlp/modules/common/megatron/megatron_utils.py | tail +190 && \
    head -1150 $PYTHON_ROOT/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py | tail +1100

# set the nemo model cache directory to mounted /data volume
ENV NEMO_CACHE_DIR=/data/models/nemo

ARG LD_PRELOAD_LIBS=""
ENV LD_PRELOAD=${LD_PRELOAD}:${LD_PRELOAD_LIBS}
