#!/usr/bin/env python3
import pyaudio
import pprint

p = pyaudio.PyAudio()

print("\nAUDIO DEVICES:\n")

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    print(f"{dev['index']:2d}: {dev['name']:50s} (inputs={dev['maxInputChannels']:<3d} outputs={dev['maxOutputChannels']:<3d} sampleRate={int(dev['defaultSampleRate'])})")
    #pprint.pprint(dev)
