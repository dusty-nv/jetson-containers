#!/usr/bin/env python3
import wave
import time
import pprint
import pyaudio
import threading

import numpy as np
import tones.mixer


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
        self.sample_clip = float(int((2 ** (self.sample_width * 8)) / 2) - 1)  # 32767 for 16-bit
        
        self.audio_chunk = 5120 #6000
        self.track_event = threading.Event()
        
        self.muted = False
 
        if self.output_file:
            self.output_wav = wave.open(self.output_file, 'wb')
            self.output_wav.setnchannels(self.channels)
            self.output_wav.setsampwidth(self.sample_width)
            self.output_wav.setframerate(self.sample_rate)
   
    def play(self, samples=None, wav=None, tone=None):
        """
        Plays either audio samples, synthesized tones/notes, or a wav file.
        Returns the audio track this is playing on.
        
        samples should be a numpy array, or int16 byte sequence
        wav should be a path to a .wav file (string)
        tone should be a dict or list of dicts (of notes)
       
        The tone/note dicts are unpacked as arguments to the following:
           https://tones.readthedocs.io/en/latest/tones.html#tones.mixer.Mixer.add_tone
           https://tones.readthedocs.io/en/latest/tones.html#tones.mixer.Mixer.add_note
           
        For example, the following would play an A-note for one second:
        
            audioMixer.play(tone={
                'note': 'a',
                'octave': 4,
                'duration': 1.0
            })
        
        You can supply a list of tones to layer multiple of them simultaneously.
        
        TODO: 
            * add gain/volume argument
            * add output channels argument
            * implement wav resampling (https://librosa.org/doc/main/generated/librosa.resample.html)
            * implement wav re-channeling
        """
        if samples is None and not wav and not tone:
            raise ValueError("either samples, wav, or tone must be specified")
            
        if wav is not None:
            with wave.open(wav, 'rb') as wav_file:
                wav_channels, wav_sample_width, wav_sample_rate, wav_num_samples, _, _ = wav_file.getparams()
                
                print(f"-- opened wav file {wav}")
                print(f"     channels:     {wav_file.getnchannels()}")
                print(f"     samples:      {wav_file.getnframes()}")
                print(f"     sample rate:  {wav_file.getframerate()}")
                print(f"     sample width: {wav_file.getsampwidth()}")
             
                samples = wav_file.readframes(wav_num_samples)
            
        if tone is not None:
            if isinstance(tone, dict):
                tone = [tone]
            elif not isinstance(tone, list):
                raise ValueError("tone should be either a dict or list of dicts")
                  
            tone_mixer = tones.mixer.Mixer(self.sample_rate, 1.0)
            
            for idx, entry in enumerate(tone):
                if not isinstance(entry, dict):
                    raise ValueError("elements of tone list should be all dicts")
                    
                tone_mixer.create_track(idx, attack=0.01, decay=0.1)
                
                if 'frequency' in entry:
                    tone_mixer.add_tone(idx, **entry)
                elif 'note' in entry:
                    tone_mixer.add_note(idx, **entry)
                else:
                    raise ValueError("tone dict should have either a 'frequency' or 'note' key")
                    
            samples = tone_mixer.sample_data()
            
        if type(samples) != np.ndarray:
            samples = np.frombuffer(samples, dtype=self.sample_type)
            
        track = {
            'status': 'playing',   # playing, paused, done
            'samples': samples,
            'playhead': 0,
            'muted': False,
        }
            
        self.tracks.append(track)
        self.track_event.set()
        return track
            
    def generate(self, in_data, frame_count, time_info, status):
        """
        Generate mixed audio samples for either the sound device, output file, web audio, ect.
        """
        samples = np.zeros(frame_count * self.channels, dtype=np.float32)

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
        
        
        samples = samples.clip(-self.sample_clip, self.sample_clip)
        samples = samples.astype(self.sample_type)

        if self.output_wav:
            self.output_wav.writeframes(samples)
            
        if self.callback:
            self.callback(samples, silent=(np.count_nonzero(samples) == 0))
            
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
                time.sleep(time_sleep * 0.9)

            """
            # this results in buffered samples (like from a wav file or TTS batch) streaming out ASAP
            # but messes up the layering of other sounds on top, that may be added after the samples
            if len(self.tracks) > 0:
                self.generate(None, self.audio_chunk, None, None)
            else:
                self.track_event.wait()
                self.track_event.clear()
            """    
            