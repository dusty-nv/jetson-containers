#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of GENAI BENCH ${GENAI_BENCH_VERSION} (branch=${GENAI_BENCH_BRANCH})"
	exit 1
fi

pip3 install genai-bench==${GENAI_BENCH_VERSION}
