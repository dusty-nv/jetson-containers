#---
# name: tensorboard
# group: ml
# depends: [python]
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV TENSORBOARD_LOG_DIR=/data/logs/tensorboard \
    TENSORBOARD_SERVER_LOGS=/data/logs/tensorboard/server.log

RUN uv pip install tensorboard gdown

CMD echo "starting tensorboard server with logs under ${TENSORBOARD_LOG_DIR}" ; \
    mkdir -p ${TENSORBOARD_LOG_DIR} ; \
    tensorboard --bind_all --logdir ${TENSORBOARD_LOG_DIR} &> ${TENSORBOARD_SERVER_LOGS} & \
    timeout 4 tail -f ${TENSORBOARD_SERVER_LOGS} ; \
    sleep 0.5 ; \
    /bin/bash
