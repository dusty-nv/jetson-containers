#---
# name: jetcam
# group: hardware
# depends: [gstreamer]
# notes: https://github.com/NVIDIA-AI-IOT/jetcam
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

#####################################
# Install traitlets
RUN echo -e "\e[48;5;172m Install traitlets \e[0m"
RUN python3 -m pip install git+https://github.com/ipython/traitlets@main

#####################################
# Install jetcam 
RUN echo -e "\e[48;5;172m Install jetcam \e[0m"
RUN python3 -m pip install git+https://github.com/NVIDIA-AI-IOT/jetcam.git@master


