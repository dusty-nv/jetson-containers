#---
# name: dli-nano-ai
# group: hardware
# depends: [pytorch, torchvision, opencv, gstreamer, nodejs, jupyter_clickable_image_widget, jetcam]
# notes: https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-RX-02+V2
#---

#####################################
# Use this jetson-containers build to create a base environment for the 
#   "Getting Started With AI on Jetson Nano" DLI course and derivative projects.
#   Course notebooks are included in the NGC containers associated with the free course:
#       https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-RX-02+V2
#       https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dli/containers/dli-nano-ai
#####################################

ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

#####################################
# Add missing libraries
RUN apt-get update \
    && apt-get -y install kmod \
    && rm -rf /var/lib/apt/lists/*

#####################################
# Set new Jupyter password and default folder
ENV JUPYTER_PASSWORD=dlinano
RUN python3 -c "from jupyter_server.auth import passwd; print(f'c.ServerApp.password = \"{passwd(\"${JUPYTER_PASSWORD}\")}\"')" > /root/.jupyter/jupyter_server_config.py
WORKDIR /nvdli-nano

#####################################
# Optional: Uncomment to pre-cache Torchvision Models 
# RUN python3 -c "import torchvision; \
#       model = torchvision.models.alexnet(weights='IMAGENET1K_V1'); \
#       model = torchvision.models.squeezenet1_1(weights='IMAGENET1K_V1'); \
#       model = torchvision.models.resnet18(weights='IMAGENET1K_V1'); \
#       model = torchvision.models.resnet34(weights='IMAGENET1K_V1')"

####################################
# Optional: Add Jupyter notebooks
# COPY notebooks .

#####################################
# Set Jupyter autostart command
CMD /bin/bash -c "jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &> /var/log/jupyter.log" & \
	echo "allow 10 sec for JupyterLab to start @ http://$(hostname -I | cut -d' ' -f1):8888 (password ${JUPYTER_PASSWORD})" && \
	echo "JupterLab logging location:  /var/log/jupyter.log  (inside the container)" && \
	/bin/bash