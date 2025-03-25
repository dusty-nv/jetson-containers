#---
# name: aim
# group: ml
# depends: python
# notes: https://github.com/aimhubio
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AIM_VERSION=3.27.0

RUN pip3 install Cython==3.0.10
RUN pip3 install aim

WORKDIR /opt/aim
RUN aim init
ENTRYPOINT ["aim"]
CMD ["up"]

COPY start-servers.sh .
#CMD ["/opt/aim/start-servers.sh"]