#!/usr/bin/env bash
set -ex

ffmpeg -version
ffmpeg -encoders
ffmpeg -decoders
ffmpeg -decoders | grep -i nvidia
