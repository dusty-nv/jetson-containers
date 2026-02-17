#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of usd_core ${USD_CORE_VERSION}"
	exit 1
fi

uv pip install usd-core==${USD_CORE_VERSION} || \
uv pip install usd-core==${USD_CORE_VERSION_SPEC}

uv pip show usd_core && python3 -c 'import usd_core'
