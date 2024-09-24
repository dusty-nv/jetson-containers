#---
# name: sapiens
# alias: sapiens
# group: graphics
# docs: docs.md
# depends: [jetson-utils, pytorch, torchvision, torchaudio, libcom, opencv:4.10.0]
# requires: '>=36'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/sapiens

RUN git clone --recursive --depth=1 https://github.com/facebookresearch/sapiens.git .

RUN pip3 install --no-cache-dir --verbose install chumpy scipy munkres tqdm cython fsspec yapf==0.40.1 matplotlib packaging omegaconf ipdb ftfy regex

RUN pip3 install --no-cache-dir --verbose json_tricks terminaltables modelindex prettytable albumentations libcom

COPY install.sh /opt/sapiens/install.sh

RUN chmod +x /opt/sapiens/install.sh && /opt/sapiens/install.sh

RUN pip3 install --force-reinstall opencv-python 'numpy<2'

COPY demo.py models.py /opt/sapiens/demo/
CMD ["/bin/bash", "-c", "export SAPIENS_ROOT=/opt/sapiens && exec /bin/bash"]