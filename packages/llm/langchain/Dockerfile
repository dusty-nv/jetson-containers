#---
# name: langchain
# group: llm
# depends: [pytorch, llama_cpp, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt
RUN pip3 install langchain[llm]

RUN pip3 install openai
RUN pip3 install llama-cpp-python

RUN mkdir langchain-notebooks
COPY *.ipynb /opt/langchain-notebooks
RUN sed 's|^\ \ }|    ,"notebook_dir": "/opt/langchain-notebooks"\n  }|' -i /root/.jupyter/jupyter_notebook_config.json

WORKDIR /