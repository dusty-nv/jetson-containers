#!/usr/bin/env python3
print('testing gstreamer-python...')

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

print('gstreamer-python OK\n')
