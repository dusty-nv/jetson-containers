#!/usr/bin/env bash
# wyoming-assist-microphone

echo "Testing microphone recording..."

arecord -D plughw:CARD=S330,DEV=0 -r 16000 -c 1 -f S16_LE -t wav -d 5 /tmp/recording.wav

echo "microphone recording OK"

echo "Playing recording..."

aplay -D plughw:CARD=S330,DEV=0 /tmp/recording.wav

echo "Playing recording OK"

rm /tmp/recording.wav
