#---
# name: tam
# group: sam
# depends: [opencv, python, numpy]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/gaomingqi/Track-Anything.git && \
    cd Track-Anything && \
    pip3 install -r requirements.txt

# Dependencies and Installation
#RUN git clone https://github.com/MCG-NKU/E2FGVI.git

CMD cd /opt/Track-Anything && python3 app.py