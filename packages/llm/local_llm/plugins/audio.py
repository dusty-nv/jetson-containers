#!/usr/bin/env python3
import logging
import pyaudio
import numpy as np

from local_llm import Plugin


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

        if self.current_buffer is None:
            if not self.input_queue.empty():
                self.current_buffer = convert_audio(self.input_queue.get())
                self.current_buffer_pos = 0
                
        if self.current_buffer is not None:
            num_samples = min(frame_count, len(self.current_buffer) - self.current_buffer_pos)
            samples[:num_samples] = self.current_buffer[self.current_buffer_pos:self.current_buffer_pos+num_samples]
            self.current_buffer_pos += num_samples
            if self.current_buffer_pos >= len(self.current_buffer):  # TODO it should loop back above for more samples
                self.current_buffer = None

        samples = samples.clip(-self.sample_clip, self.sample_clip)
        samples = samples.astype(self.sample_type)

        return (samples, pyaudio.paContinue)


def convert_audio(samples, dtype=np.int16):
    """
    Convert between audio datatypes like float<->int16 and apply sample re-scaling
    """
    if samples.dtype == dtype:
        return samples
        
    sample_width = np.dtype(dtype).itemsize
    max_value = float(int((2 ** (sample_width * 8)) / 2) - 1)  # 32767 for 16-bit
        
    if samples.dtype == np.float32 or samples.dtype == np.float64:  # float-to-int
        samples = samples * max_value
        samples = samples.clip(-max_value, max_value)
        samples = samples.astype(dtype)
    elif dtype == np.float32 or dtype == np.float64:  # int-to-float
        samples = samples.astype(dtype)
        samples = samples / max_value
    else:
        raise TypeError(f"unsupported audio sample dtype={samples.dtype}")
        
    return samples
        
        
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
    
    audio_output = AudioOutputDevice(**vars(args)).start()
    
    audio_output.input(samples)
    time.sleep(args.length + 1.0)
    
    