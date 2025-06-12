#---
# name: protobuf:apt
# group: build
# depends: [python]
# test: [test.py, test.sh]
# notes: install protobuf from apt repo
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# confirm first that the protobuf:cpp package isn't already installed
RUN if [ -z "$(which protoc)" ]; then \
        apt-get update; \
	   apt-get install -y --no-install-recommends \
	       protobuf-compiler \
		  libprotoc-dev \
		  libprotobuf-dev; \
	   rm -rf /var/lib/apt/lists/*; \
	   apt-get clean; \
	   pip3 install --upgrade  protobuf; \
    fi

