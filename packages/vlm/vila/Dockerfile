#---
# name: vila
# group: vlm
# depends: [pytorch, flash-attention, transformers, opencv, deepspeed, decord]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/NVlabs/VILA/git/refs/heads/main /tmp/vila_version.json

RUN git clone --branch=main --depth=1 --recursive https://github.com/NVlabs/VILA /opt/VILA && \
    cd /opt/VILA && \
    sed -i 's|"torch.*"|"torch"|' pyproject.toml && \
    sed -i 's|"pytorchvideo.*"|"pytorchvideo"|' pyproject.toml && \
    sed -i 's|"opencv-python.*"|"opencv-python"|' pyproject.toml && \
    sed -i 's|"bitsandbytes.*"|"bitsandbytes"|' pyproject.toml && \
    sed -i 's|"deepspeed.*"|"deepspeed"|' pyproject.toml && \
    sed -i 's|"sentencepiece.*"|"sentencepiece"|' pyproject.toml && \
    sed -i 's|"scikit-learn.*"|"scikit-learn"|' pyproject.toml && \
    sed -i 's|"accelerate.*"|"accelerate"|' pyproject.toml && \
    sed -i 's|"decord.*"|"decord"|' pyproject.toml && \
    sed -i 's|"datasets.*"|"datasets"|' pyproject.toml && \
    sed -i 's|"numpy.*"|"numpy"|' pyproject.toml && \
    sed -i 's|"nltk.*"|"nltk"|' pyproject.toml && \
    sed -i 's|"timm.*"|"timm"|' pyproject.toml && \
    cat pyproject.toml && \
    pip3 install -e . && \
    pip3 install -e ".[train]" && \
    pip3 install -e ".[eval]" 
    
#RUN pip3 install 'transformers==4.37.2' && \
#    site_pkg_path=$(python3 -c 'import site; print(site.getsitepackages()[0])') && \
#    cp -rv /opt/VILA/llava/train/transformers_replace/* $site_pkg_path/transformers/ && \
#    cp -rv /opt/VILA/llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

