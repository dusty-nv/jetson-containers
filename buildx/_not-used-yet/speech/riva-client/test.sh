#!/usr/bin/env bash
RIVA_BIN=/opt/riva/cpp-clients/bazel-bin/riva/clients

$RIVA_BIN/asr/riva_asr_client --help
$RIVA_BIN/nlp/riva_nlp_qa --help
$RIVA_BIN/tts/riva_tts_client --help

exit 0