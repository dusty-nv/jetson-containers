#!/usr/bin/env bash
# wyoming-assist-microphone

echo "RECORDING DEVICES:"
arecord -L

echo "PLAYING DEVICES:"
aplay -L
