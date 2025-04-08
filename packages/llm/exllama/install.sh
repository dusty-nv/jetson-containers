#!/usr/bin/env bash
set -ex

git clone --branch=v${EXLLAMA_BRANCH} --depth=1 --recursive https://github.com/turboderp/exllamav3 /opt/exllamav3

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of exllamav3 ${EXLLAMA_VERSION} (branch=${EXLLAMA_VERSION})"
	exit 1
fi

pip3 install exllamav3==${EXLLAMA_VERSION}

python3 -c 'import exllamav3; print(exllamav3.__version__);'