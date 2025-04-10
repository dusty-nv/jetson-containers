#!/usr/bin/env python3
import time
import wave
import logging
import pyaudio
import numpy as np

from local_llm import Plugin
from local_llm.utils import convert_audio


class AudioOutputDevice(Plugin):
    """
    Output audio to an audio interface attached to the machine.
    Expects to recieve audio samples as input, np.ndarray with dtype=float or int16
    """
    def __init__(self, audio_output_device=0, audio_output_channels=1, sample_rate_hz=48000, **kwargs):
        """
        Parameters:
        
          audio_output_device (int) -- audio output device number for PortAudio
          audio_output_channels (int) -- 1 for mono, 2 for stereo
          sample_rate_hz (int) -- sample rate of any outgoing audio (typically 16000, 44100, 48000)
        """
        super().__init__(**kwargs)
        
        self.pa = pyaudio.PyAudio()
        
        self.output_device_id = audio_output_device
        self.output_device = None
        
        self.sample_rate = sample_rate_hz
        self.sample_type = np.int16
        self.sample_width = 2
        self.sample_clip = float(int((2 ** (self.sample_width * 8)) / 2) - 1)  # 32767 for 16-bit
        
        self.paused = -1
        self.channels = audio_output_channels
        self.current_buffer = None
        self.current_buffer_pos = 0
        
        
    def run(self):
        """
        Open the audio output interface, which will call generate_audio() from another thread
        """
        self.output_device = self.pa.open(
                output_device_index=self.output_device_id,
                format=self.pa.get_format_from_width(self.sample_width),
                channels=self.channels,
                rate=self.sample_rate,
                stream_callback=self.generate,
                output=True,
            )        
    
    def generate(self, in_data, frame_count, time_info, status):
        """
        Generate mixed audio samples for either the sound device, output file, web audio, ect.
        """
        samples = np.zeros(frame_count * self.channels, dtype=self.sample_type)
        samples_idx = 0

        if self.interrupted:
            self.current_buffer = None
            self.interrupted = False
            
        while not self.is_paused():
            if self.current_buffer is None:
                if not self.input_queue.empty():
                    self.current_buffer = convert_audio(self.input_queue.get()[0], dtype=self.sample_type)
                    self.current_buffer_pos = 0
                    
            if self.current_buffer is not None:
                num_samples = min(frame_count-samples_idx, len(self.current_buffer) - self.current_buffer_pos)
                samples[samples_idx:samples_idx+num_samples] = self.current_buffer[self.current_buffer_pos:self.current_buffer_pos+num_samples]
                self.current_buffer_pos += num_samples
                samples_idx += num_samples
                if self.current_buffer_pos >= len(self.current_buffer):  # TODO it should loop back above for more samples
                    self.current_buffer = None
                    continue
                    
            break
            
        samples = samples.clip(-self.sample_clip, self.sample_clip)
        samples = samples.astype(self.sample_type)

        return (samples, pyaudio.paContinue)

    def pause(self, duration=0):
        """
        Pause audio playback for `duration` number of seconds
        If `duration` is 0, it will be paused until unpaused.
        If `duration` is negative, it will be unpaused.
        """
        if duration <= 0:
            self.paused = duration
        else:
            self.paused = time.perf_counter() + duration
            logging.debug(f"pausing audio output for {duration} seconds")
            
    def unpause(self):
        """
        Unpause audio playback
        """
        self.pause(-1.0)
        
    def is_paused(self):
        """
        Returns true if playback is currently paused.
        """
        if self.paused < 0:
            return False
            
        if self.paused > 0 and time.perf_counter() >= self.paused:
            self.paused = -1
            return False
            
        return True

        
class AudioOutputFile(Plugin):
    """
    Output audio to a wav file
    Expects to recieve audio samples as input, np.ndarray with dtype=float or int16
    TODO:  this doesn't fill in gaps for "realtime playback"
    """
    def __init__(self, audio_output_file='output.wav', audio_output_channels=1, sample_rate_hz=48000, **kwargs):
        """
        Parameters:
        
          audio_output_file (str) -- path to the output wav file
          audio_output_channels (int) -- 1 for mono, 2 for stereo
          sample_rate_hz (int) -- sample rate of any outgoing audio (typically 16000, 44100, 48000)
        """
        super().__init__(**kwargs)
        
        self.pa = pyaudio.PyAudio()
        
        self.output_file = audio_output_file
        self.channels = audio_output_channels
        
        self.sample_type = np.int16
        self.sample_width = np.dtype(self.sample_type).itemsize
        self.sample_clip = float(int((2 ** (self.sample_width * 8)) / 2) - 1)  # 32767 for 16-bit
        self.sample_rate = sample_rate_hz
        
        self.wav = wave.open(self.output_file, 'wb')
        
        self.wav.setnchannels(self.channels)
        self.wav.setsampwidth(self.sample_width)
        self.wav.setframerate(self.sample_rate)
    
    def process(self, input, **kwargs):
        """
        Save float or int16 audio samples to wav file
        TODO: append silence before the last number of samples written is less in duration
              than the time since process() was last called
        """
        input = convert_audio(input, dtype=self.sample_type)
        self.wav.writeframes(input)
        
        
if __name__ == "__main__":
    import time
    from local_llm.utils import ArgParser

    parser = ArgParser(extras=['audio_output', 'log'])
    
    parser.add_argument('--frequency', type=float, default=440)
    parser.add_argument('--length', type=float, default=2.0)
    parser.add_argument('--volume', type=float, default=1.0)
    
    args = parser.parse_args()
    
    def noteGenerator(frequency=440, length=0.2, amplitude=1, sample_rate=44100):
        timepoints = np.linspace(0, length, int(length*sample_rate))
        data = amplitude*np.sin(2*np.pi*frequency*timepoints)  # A*sin(2*π*f*t)
        return data
        
    samples = noteGenerator(args.frequency, args.length, args.volume, args.sample_rate_hz)

    print(samples)
    print(samples.shape)
    
    if args.audio_output_device is not None:
        device = AudioOutputDevice(**vars(args)).start()
        device.input(samples)
        
    if args.audio_output_file is not None:
        file = AudioOutputFile(**vars(args)).start()
        file.input(samples)
        
    time.sleep(args.length + 1.0)
