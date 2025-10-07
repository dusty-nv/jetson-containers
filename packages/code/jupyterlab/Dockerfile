#---
# name: jupyterlab
# group: code
# depends: [cuda, cudastack:standard, cuda-python, pycuda, rust]
# docs: docs.md
# notes: will autostart Jupyter server on port 8888 unless container entry CMD is overridden
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# set a specific version to install like '==4.2.0'
# (or by default leaving it blank will install the latest)
ARG JUPYTERLAB_VERSION_SPEC=""

# https://stackoverflow.com/questions/61553491/how-to-make-jupyter-terminal-fully-functional
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]

# enable colors in the jupyterlab terminal
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# jupyterlab<4 -- ModuleNotFoundError: No module named 'notebook.auth'
RUN uv pip install jupyter "jupyterlab${JUPYTERLAB_VERSION_SPEC}" && \
    uv pip install jupyterlab_widgets

RUN uv pip list
RUN jupyter --version
RUN jupyter labextension list

RUN jupyter lab --version && jupyter lab --generate-config
# RUN python3 -c "from notebook.auth.security import set_password; set_password('nvidia', '/root/.jupyter/jupyter_notebook_config.json')"

COPY jupyter_server_config.json /root/.jupyter/
COPY start_jupyter /
CMD /start_jupyter && /bin/bash

