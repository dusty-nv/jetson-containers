#!/usr/bin/env bash

set -ex

echo "Testing piper-phonemize..."

${PIPER_PHONEMIZE_DIR}/piper_phonemize -l en-us --espeak-data ${ESPEAK_NG_DATA_DIR} <<EOF
This is a test.
This is another test!
EOF

echo "piper-phonemize OK"
