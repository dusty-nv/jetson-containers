#!/usr/bin/env python3
# flask webserver (run automatically by chat.py)
import sys
import ssl
import time
import json
import flask
import queue
import base64
import pprint
import asyncio
import threading

import wave

from websockets.sync.server import serve as websocket_serve


class Webserver(threading.Thread):
    """
    Flask + websockets server for the chat interface
    """
    def __init__(self, web_server='0.0.0.0', web_port=8050, ws_port=49000, ssl_cert=None, ssl_key=None, 
                 text_callback=None, audio_callback=None, **kwargs):
                 
        super(Webserver, self).__init__(daemon=True)  # stop thread on main() exit
        
        self.host = web_server
        self.port = web_port
        
        self.text_callback = text_callback
        self.audio_callback = audio_callback
        
        # SSL / HTTPS
        self.ssl_key = ssl_key
        self.ssl_cert = ssl_cert
        self.ssl_context = None
        
        if self.ssl_cert and self.ssl_key:
            #self.ssl_context = (self.ssl_cert, self.ssl_key)
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(certfile=self.ssl_cert, keyfile=self.ssl_key)
            
        # flask server
        self.app = flask.Flask(__name__)
        self.app.add_url_rule('/', view_func=self.on_index, methods=['GET'])
        
        # websocket
        self.ws_port = ws_port
        self.ws_queue = queue.Queue()
        self.ws_server = websocket_serve(self.on_websocket, host=self.host, port=self.ws_port, ssl_context=self.ssl_context)
        self.ws_thread = threading.Thread(target=lambda: self.ws_server.serve_forever(), daemon=True)
       
    @staticmethod
    def on_index():
        return flask.render_template('index.html')
            
    def on_websocket(self, websocket):
        print(f"-- new websocket connection from {websocket.remote_address}")

        listener_thread = threading.Thread(target=self.websocket_listener, args=[websocket], daemon=True)
        listener_thread.start()
 
        # empty the queue from before the connection was made
        # (otherwise client will be flooded with old messages)
        # TODO implement self.connected so the ws_queue doesn't grow so large without webclient connected...
        while True:
            try:
                self.ws_queue.get(block=False)
            except queue.Empty:
                break
            
        while True:
            websocket.send(json.dumps(self.ws_queue.get()))

    def websocket_listener(self, websocket):
        print(f"-- listening on websocket connection from {websocket.remote_address}")

        #wav = wave.open('/data/audio/capture.wav', 'wb')
        #wav.setnchannels(1)
        #wav.setsampwidth(2)
        #wav.setframerate(48000)
            
        while True:
            msg = websocket.recv()
            msg = json.loads(msg)
            
            if msg['type'] == 'audio':
                #print(f"recieved audio:  size={msg['size']}")
                #print(msg['settings']);
                msg['data'] = base64.b64decode(msg['data'])
                if len(msg['data']) != msg['size']:
                    raise RuntimeError(f"web audio chunk had length={len(msg['data'])}  (expected={msg['size']})")
                #wav.writeframesraw(msg['data'])
                #wav.writeframes(msg['data'])
                if self.audio_callback:
                    self.audio_callback(msg)
    
    def send_message(self, msg):
        self.ws_queue.put(msg)   # do we even need this queue at all and can the websocket just send straight away?
        
    def output_audio(self, samples):
        self.send_message({
            'type': 'audio',
            'data': samples.tolist()
        })

    def run(self):
        print(f"-- starting webserver @ {self.host}:{self.port}")
        self.ws_thread.start()
        self.app.run(host=self.host, port=self.port, ssl_context=self.ssl_context, debug=True, use_reloader=False)
        