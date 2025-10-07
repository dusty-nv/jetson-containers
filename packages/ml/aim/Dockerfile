#---
# name: aim
# group: ml
# depends: python
# notes: https://github.com/aimhubio
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AIM_VERSION=3.28.0

RUN uv pip install Cython==3.0.10
RUN uv pip install aim

WORKDIR /opt/aim
RUN aim init
ENTRYPOINT ["aim"]
CMD ["up"]

COPY start-servers.sh .
#CMD ["/opt/aim/start-servers.sh"]
