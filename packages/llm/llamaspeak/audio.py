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
    def __init__(self, output_device=None, output_file=None, callback=None, sample_rate_hz=44100, audio_channels=1, **kwargs):
                 
        super(AudioMixer, self).__init__(daemon=True)
        
        self.pa = pyaudio.PyAudio()

        self.tracks = []
        self.channels = audio_channels
        self.callback = callback
        self.output_device_id = output_device
        self.output_device = None
        self.output_wav = None
        self.output_file = output_file
        self.sample_rate = sample_rate_hz
        self.sample_type = np.int16
        self.sample_width = 2
        self.audio_chunk = 6000
        self.muted = False
 
        if self.output_file:
            self.output_wav = wave.open(self.output_file, 'wb')
            self.output_wav.setnchannels(self.channels)
            self.output_wav.setsampwidth(self.sample_width)
            self.output_wav.setframerate(self.sample_rate)

     
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
            
    def generate(self, in_data, frame_count, time_info, status):
        """
        Generate mixed audio samples for either the sound device, output file, web audio, ect.
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
        
        #samples = samples.tobytes()
        
        if self.output_wav:
            self.output_wav.writeframes(samples) #raw(samples)
            
        if self.callback:
            self.callback(samples)
            
        return (samples, pyaudio.paContinue)
        
    def run(self):
        """
        Used when there's not a soundcard output device
        """
        if self.output_device_id is not None:
            print(f"-- opening audio output device ({self.output_device_id})")
            self.output_device = self.pa.open(
                output_device_index=self.output_device_id,
                format=self.pa.get_format_from_width(self.sample_width),
                channels=self.channels,
                rate=self.sample_rate,
                stream_callback=self.generate,
                output=True,
            )
            return # the soundcard callback will run in another thread
            
        if not self.output_wav and not self.callback:
            return

        print(f"-- running AudioMixer thread")
        
        while True:
            time_start = time.perf_counter()
            self.generate(None, self.audio_chunk, None, None)
            time_sleep = (self.audio_chunk / self.sample_rate) - (time.perf_counter() - time_start)
            
            if time_sleep > 0.001:
                #print(f"-- AudioMixer sleeping for {time_sleep} seconds")
                time.sleep(time_sleep)
                
            