#!/usr/bin/env bash
# wyoming-assist-microphone

echo "Testing microphone recording..."

arecord -D plughw:CARD=S330,DEV=0 -r 16000 -c 1 -f S16_LE -t wav -d 5 /tmp/recording.wav

echo "microphone recording OK"

echo "Testing speaker..."

aplay -D plughw:CARD=S330,DEV=0 /tmp/recording.wav

echo "Speaker OK"

rm /tmp/recording.wav
