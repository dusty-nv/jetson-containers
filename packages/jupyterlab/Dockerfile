#---
# name: jupyterlab
# group: core
# depends: [python, numpy, rust]
# notes: will autostart Jupyter server on port 8888 unless container entry CMD is overridden
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# jupyterlab<4 -- ModuleNotFoundError: No module named 'notebook.auth'
RUN pip3 install --no-cache-dir --verbose jupyter 'jupyterlab<4' && \
    pip3 install --no-cache-dir --verbose jupyterlab_widgets
    
RUN jupyter lab --version && jupyter lab --generate-config
RUN python3 -c "from notebook.auth.security import set_password; set_password('nvidia', '/root/.jupyter/jupyter_notebook_config.json')"

CMD /bin/bash -c "jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &> /var/log/jupyter.log" & \
	echo "allow 10 sec for JupyterLab to start @ http://$(hostname -I | cut -d' ' -f1):8888 (password nvidia)" && \
	echo "JupterLab logging location:  /var/log/jupyter.log  (inside the container)" && \
	/bin/bash
