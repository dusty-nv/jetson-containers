#---
# name: vila
# group: vlm
# depends: [pytorch, flash-attention, transformers, opencv, deepspeed, decord2]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/NVlabs/VILA/git/refs/heads/main /tmp/vila_version.json

RUN git clone --branch=main --depth=1 --recursive https://github.com/NVlabs/VILA /opt/VILA && \
    cd /opt/VILA && \
    sed 's|"bitsandbytes.*",||' -i pyproject.toml && \
    sed -i 's/==/>=/g' pyproject.toml && \
    sed -i 's/~=/>=/g' pyproject.toml && \
    cat pyproject.toml && \
    pip3 install -e . && \
    pip3 install -e ".[train]" && \
    pip3 install -e ".[eval]"

#RUN pip3 install 'transformers==4.37.2' && \
#    site_pkg_path=$(python3 -c 'import site; print(site.getsitepackages()[0])') && \
#    cp -rv /opt/VILA/llava/train/transformers_replace/* $site_pkg_path/transformers/ && \
#    cp -rv /opt/VILA/llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

