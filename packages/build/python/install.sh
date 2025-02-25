#!/usr/bin/env bash
# Python installer
set -x

apt-get update
apt-get install -y --no-install-recommends \
	python${PYTHON_VERSION} \
	python${PYTHON_VERSION}-dev

which python${PYTHON_VERSION}
return_code=$?
set -e

if [ $return_code != 0 ]; then
   echo "-- using deadsnakes ppa to install Python ${PYTHON_VERSION}"
   add-apt-repository ppa:deadsnakes/ppa
   apt-get update
   apt-get install -y --no-install-recommends \
	  python${PYTHON_VERSION} \
	  python${PYTHON_VERSION}-dev
fi

# path 1:  Python 3.8-3.10 for JP5/6
# path 2:  Python 3.6 for JP4
# path 3:  Python 3.12 for 24.04
curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} || \
curl -sS https://bootstrap.pypa.io/pip/3.6/get-pip.py | python3.6 || \
apt-get install -y --no-install-recommends python3-venv && \
python3 -m venv ${VIRTUAL_ENV} && source ${VIRTUAL_ENV}/bin/activate && \
curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

rm -rf /var/lib/apt/lists/*
apt-get clean

ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3
#ln -s /usr/bin/pip${PYTHON_VERSION} /usr/local/bin/pip3

# this was causing issues downstream (e.g. Python2.7 still around in Ubuntu 18.04, \
# and in cmake python enumeration where some packages expect that 'python' is 2.7) \
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \  \
#    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \

which python3
python3 --version

which pip3
pip3 --version

python3 -m pip install --upgrade pip pkginfo --index-url https://pypi.org/simple

pip3 install --no-cache-dir --verbose --no-binary :all: psutil
pip3 install --upgrade --no-cache-dir \
   setuptools \
   packaging \
   'Cython' \
   wheel 

pip3 install --upgrade --no-cache-dir --index-url https://pypi.org/simple \
   twine
   
