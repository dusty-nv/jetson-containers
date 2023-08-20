#!/usr/bin/env python3
import wave
import time
import pprint
import threading
import pyaudio

import numpy as np


class AudioMixer(threading.Thread):
    """
    Multi-track audio output mixer / sound generator
    """
    def __init__(self, output_device=None, output_file=None, sample_rate_hz=44100, audio_channels=1, **kwargs):
                 
        super(AudioMixer, self).__init__(daemon=True)
        
        self.pa = pyaudio.PyAudio()

        self.tracks = []
        self.channels = audio_channels
        self.output_device = None
        self.output_wav = None
        self.output_file = output_file
        self.sample_rate = sample_rate_hz
        self.sample_type = np.int16
        self.sample_width = 2
        self.muted = False
        self.opened = False

        if output_device is not None:
            self.output_device = self.pa.open(
                output_device_index=self.output_device,
                format=self.pa.get_format_from_width(self.sample_width),
                channels=self.channels,
                rate=self.sample_rate,
                stream_callback=self.need_audio,
                output=True,
            )
            self.opened = True
   
        if self.output_file:
            self.output_wav = wave.open(self.output_file, 'wb')
            self.output_wav.setnchannels(self.channels)
            self.output_wav.setsampwidth(self.sample_width)
            self.output_wav.setframerate(self.sample_rate)
            self.opened = True
            
            if self.output_device is None:
                self.start()
     
    def play(self, samples=None, filename=None, tone=None):
        """
        Plays either audio samples, a wav file, or a tone.
        Returns the audio track this is playing on.
        TODO: only 'samples' mode is implemented
        """
        if samples is None and not filename and not tone:
            raise ValueError("either samples, filename, or tone must be specified")
            
        if type(samples) != np.ndarray:
            samples = np.frombuffer(samples, dtype=self.sample_type)
            
        track = {
            'status': 'playing',   # playing, paused, done
            'samples': samples,
            'playhead': 0,
            'muted': False,
        }
            
        self.tracks.append(track)
        return track
            
    def need_audio(self, in_data, frame_count, time_info, status):
        """
        Callback from the sound device when it needs audio samples to output
        """
        samples = np.zeros(frame_count * self.channels, dtype=self.sample_type)
        
        for track in self.tracks.copy():
            playhead = track['playhead']
            num_samples = min(frame_count, len(track['samples']) - playhead)
            
            if not self.muted and track['status'] == 'playing' and not track['muted']:
                samples[:num_samples] += track['samples'][playhead:playhead+num_samples]
                
            if track['status'] != 'paused':
                track['playhead'] += frame_count
            
            if track['playhead'] >= len(track['samples']):
                track['status'] = 'done'
                self.tracks.remove(track)
        
        if self.output_wav:
            self.output_wav.writeframesraw(samples)
            
        return (samples, pyaudio.paContinue)
        
    def run(self):
        """
        Thread main (only gets used for wav-only output)
        """
        print(f"-- running AudioMixer thread")
        
        if not self.output_wav:
            return
            
        while True:
            self.output_wav.writeframesraw(
                self.need_audio(None, self.sample_rate_hz, None, None)
            )
            time.sleep(1.0)