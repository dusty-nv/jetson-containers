#---
# name: sapiens
# group: vit
# depends: [jetson-utils, pytorch, torchvision, torchaudio, libcom, opencv]
# requires: '>=36'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/sapiens

RUN git clone --recursive --depth=1 https://github.com/facebookresearch/sapiens.git .

RUN uv pip install install chumpy scipy munkres tqdm cython fsspec yapf==0.40.1 matplotlib packaging omegaconf ipdb ftfy regex

RUN uv pip install json_tricks terminaltables modelindex prettytable albumentations libcom

COPY install.sh /opt/sapiens/install.sh

RUN chmod +x /opt/sapiens/install.sh && /opt/sapiens/install.sh

RUN uv pip install --force-reinstall opencv-python

COPY demo.py models.py /opt/sapiens/demo/
CMD ["/bin/bash", "-c", "export SAPIENS_ROOT=/opt/sapiens && exec /bin/bash"]
