#---
# name: dli-nano-ai
# group: hardware
# depends: [pytorch, torchvision, opencv, gstreamer, nodejs, jupyter_clickable_image_widget, jetcam]
# notes: https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-RX-02+V2
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# ENV DEBIAN_FRONTEND=noninteractive
ENV JUPYTER_PASSWORD=dlinano

WORKDIR /nvdli-nano

#####################################
# Pre-cache Torchvision Models 
RUN python3 -c "import torchvision; \
                model = torchvision.models.alexnet(weights='IMAGENET1K_V1'); \
                model = torchvision.models.squeezenet1_1(weights='IMAGENET1K_V1'); \
                model = torchvision.models.resnet18(weights='IMAGENET1K_V1'); \
                model = torchvision.models.resnet34(weights='IMAGENET1K_V1')"

#####################################
# Set Jupyter password
RUN rm -rf /root/.jupyter && \
jupyter lab --version && jupyter lab --generate-config && \
python3 -c "from notebook.auth.security import set_password; set_password('${JUPYTER_PASSWORD}', '/root/.jupyter/jupyter_notebook_config.json')"

#####################################
# Fix initial Gstreamer warnings
#   (work-around per https://forums.developer.nvidia.com/t/there-are-some-gstream-warnings-after-installing-deepstream-6-0/194104/4?u=dsheahen)
# COPY configs/gstreamer_configs/ /root/.cache/

#####################################
# Add Jupyter notebooks
# COPY notebooks .

#####################################
# Set Jupyter autostart command
CMD /bin/bash -c "jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &> /var/log/jupyter.log" & \
	echo "allow 10 sec for JupyterLab to start @ http://$(hostname -I | cut -d' ' -f1):8888 (password ${JUPYTER_PASSWORD})" && \
	echo "JupterLab logging location:  /var/log/jupyter.log  (inside the container)" && \
	/bin/bash
    

