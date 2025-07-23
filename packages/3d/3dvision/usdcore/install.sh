#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of usd_core ${USD_CORE_VERSION}"
	exit 1
fi

pip3 install usd-core==${USD_CORE_VERSION} || \
pip3 install usd-core==${USD_CORE_VERSION_SPEC}

pip3 show usd_core && python3 -c 'import usd_core'
