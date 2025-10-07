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
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            nodejs nodejs-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean
RUN npm install -g node-gyp
# RUN uv pip install jupyter 'jupyterlab==4.2.0' && \
# RUN uv pip install jupyter 'jupyterlab<4' && \
#     uv pip install jupyterlab_widgets
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# For ipywidget development
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#             cookiecutter && \
#     rm -rf /var/lib/apt/lists/* && \
#     apt-get clean
# RUN nmp install -g yarn
# RUN uv pip install jupyter_packaging

WORKDIR /opt

# Install jupyter_clickable_image_widget
RUN echo "\e[42m Install jupyter_clickable_image_widget \e[0m"
RUN git clone https://github.com/jaybdub/jupyter_clickable_image_widget && \
    cd jupyter_clickable_image_widget && \
    git checkout b2ee0f4   && \
# RUN git clone https://github.com/tokk-nv/jupyter_clickable_image_widget/ && \
#     cd jupyter_clickable_image_widget && \
#     git checkout dev-ipywidgets8 && \
    uv pip install . && \
    jupyter labextension install js && \
    jupyter lab build
# COPY ipyclick_tokknv /opt/jupyter_clickable_image_widget
# RUN cd /opt/jupyter_clickable_image_widget && \
#     uv pip install . && \
#     jupyter labextension install js && \
#     jupyter lab build

# RUN uv pip install 'ipywidgets<8'
RUN uv pip list
RUN jupyter --version
RUN jupyter labextension list

COPY ./examples/ /opt/jupyter_clickable_examples/
ENV JUPYTER_ROOT=/opt/jupyter_clickable_examples/

CMD /start_jupyter && /bin/bash

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
