#
# upgrade cmake with apt
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


# upgrade cmake - https://stackoverflow.com/a/56690743
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  software-properties-common \
		  apt-transport-https \
		  ca-certificates \
		  gnupg \
		  lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
		  
# typically --only-upgrade is used in the apt install, but cmake wasn't previously installed in this container	
# previous issue regarding kitware's rotating GPG keys: https://github.com/dusty-nv/jetson-containers/issues/216
RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get update && \
    rm /usr/share/keyrings/kitware-archive-keyring.gpg && \
    apt-get install -y --no-install-recommends kitware-archive-keyring \
    apt-cache policy cmake && \
    apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN cmake --version
