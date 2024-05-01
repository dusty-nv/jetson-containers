#!/usr/bin/env bash
set -ex

git clone --branch=v${EXLLAMA_BRANCH} --depth=1 --recursive https://github.com/turboderp/exllamav2 /opt/exllamav2

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of exllamav2 ${EXLLAMA_VERSION} (branch=${EXLLAMA_VERSION})"
	exit 1
fi

pip3 install --no-cache-dir --verbose exllamav2==${EXLLAMA_VERSION}

python3 -c 'import exllamav2; print(exllamav2.__version__);'