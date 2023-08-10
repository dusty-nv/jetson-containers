#!/usr/bin/env python3
print('testing riva.client (Python)')

import riva.client

print('riva.client version:', riva.client.__version__)

from riva.client import ASRService
from riva.client import NLPService
from riva.client import SpeechSynthesisService

print('riva.client (Python) OK\n')
