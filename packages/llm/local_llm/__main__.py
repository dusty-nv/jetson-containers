#!/usr/bin/env python3
# the terminal-based chat program was moved to chat/__main__.py
# forward deprecated 'python3 -m local_llm' calls to local_llm.chat
import runpy
#import logging

#logging.warning("'python3 -m local_llm' is deprecated, please run local_llm.chat or local_llm.completion instead")

runpy.run_module('local_llm.chat', run_name='__main__')
