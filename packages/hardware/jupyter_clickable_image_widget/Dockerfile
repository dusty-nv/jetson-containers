#---
# name: jupyter_clickable_image_widget
# group: hardware
# depends: [jupyterlab]
# notes: https://github.com/jaybdub/jupyter_clickable_image_widget
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Reference:
# https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docker/base/Dockerfile


# Install traitlets (master, to support the unlink() method)
RUN echo -e "\e[48;5;172m Install traitlets \e[0m"
RUN python3 -m pip install git+https://github.com/ipython/traitlets@main

# Install dependencies for jupyter_clickable_image_widget
RUN echo "\e[42m Install dependencies \e[0m"
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            libssl-dev \
            nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

WORKDIR /opt

# Install jupyter_clickable_image_widget
RUN echo "\e[42m Install jupyter_clickable_image_widget \e[0m"
RUN git clone https://github.com/jaybdub/jupyter_clickable_image_widget && \
    cd jupyter_clickable_image_widget && \
    git checkout tags/v0.1 && \
    pip3 install . && \
    jupyter labextension install js && \
    jupyter lab build

RUN pip3 install 'ipywidgets<8'

# {
#   "name": "jupyter_clickable_image_widget",
#   "version": "0.1.0",
#   "description": "A Custom Jupyter Widget Library",
#   "author": "John Welsh",
#   "main": "lib/index.js",
#   "repository": {
#     "type": "git",
#     "url": "https://github.com//jupyter_clickable_image_widget.git"
#   },
#   "keywords": [
#     "jupyter",
#     "widgets",
#     "ipython",
#     "ipywidgets",
#     "jupyterlab-extension"
#   ],
#   "files": [
#     "lib/**/*.js",
#     "dist/*.js"
#   ],
#   "scripts": {
#     "clean": "rimraf dist/",
#     "prepublish": "webpack",
#     "build": "webpack",
#     "watch": "webpack --watch --mode=development",
#     "test": "echo \"Error: no test specified\" && exit 1"
#   },
#   "devDependencies": {
#     "webpack": "^3.5.5",
#     "rimraf": "^2.6.1"
#   },
#   "dependencies": {
#     "@jupyter-widgets/base": "^1.1 || ^2 || ^3 || ^4 || ^6"
#   },
#   "jupyterlab": {
#     "extension": "lib/labplugin"
#   }
# }