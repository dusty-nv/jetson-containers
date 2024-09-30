#---
# name: diffusion_policy
# alias: diffusion-policy
# group: robots
# docs: docs.md
# depends: [transformers, robomimic, opencv, numba, jupyterlab]
# requires: '>=36'
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN git clone --depth=1 https://github.com/real-stanford/diffusion_policy /opt/diffusion_policy
COPY requirements.txt /opt/diffusion_policy/

RUN pip3 install --no-cache-dir --verbose -r /opt/diffusion_policy/requirements.txt
RUN pip3 install --no-cache-dir --verbose git+https://github.com/facebookresearch/pytorch3d.git
RUN pip3 install --no-cache-dir --verbose 'protobuf==3.20.2' 'accelerate'
RUN pip3 install --verbose -e /opt/diffusion_policy 

COPY *.ipynb /opt/diffusion_policy/notebooks/

CMD /bin/bash