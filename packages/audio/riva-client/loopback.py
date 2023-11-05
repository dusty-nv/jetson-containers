#!/usr/bin/env python3
# Riva test for feeding ASR from microphone into TTS
import argparse
import time
import wave

import riva.client
import riva.client.audio_io

from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters


def parse_args():
    default_device_info = riva.client.audio_io.get_default_input_device_info()
    default_device_index = None if default_device_info is None else default_device_info['index']
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--list-devices", action="store_true", help="List output audio devices indices.")
    parser.add_argument("--input-device", type=int, default=default_device_index, help="An input audio device to use.")
    parser.add_argument("--output-device", type=int, help="Output device to use.")
    parser.add_argument("-o", "--output", type=str, help="Output file .wav file to write synthesized audio.")
    
    parser.add_argument("--no-punctuation", action='store_true', help="Disable ASR automatic punctuation")
    parser.add_argument("--voice", help="A voice name to use for TTS. If this parameter is missing, then the server will try a first available model based on parameter `--language-code`.")
    
    parser.add_argument("--sample-rate-hz", type=int, default=16000, help="Number of audio frames per second in synthesized audio.")
    parser.add_argument("--file-streaming-chunk", type=int, default=1600, help="A maximum number of frames in a audio chunk sent to server.")

    parser = add_asr_config_argparse_parameters(parser, profanity_filter=True)
    parser = add_connection_argparse_parameters(parser)
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    
    if args.list_devices:
        riva.client.audio_io.list_output_devices()
        return
        
    auth = riva.client.Auth(args.ssl_cert, args.use_ssl, args.server)
    
    nchannels = 1
    sampwidth = 2
    
    asr_service = riva.client.ASRService(auth)
    tts_service = riva.client.SpeechSynthesisService(auth)
    
    asr_config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code=args.language_code,
            max_alternatives=1,
            profanity_filter=args.profanity_filter,
            enable_automatic_punctuation=not args.no_punctuation,
            verbatim_transcripts=not args.no_verbatim_transcripts,
            sample_rate_hertz=args.sample_rate_hz,
            audio_channel_count=nchannels,
        ),
        interim_results=True,
    )
    
    riva.client.add_word_boosting_to_config(asr_config, args.boosted_lm_words, args.boosted_lm_score)

    speaker_stream, wav_out = None, None
    
    try:
        if args.output_device is not None:
            speaker_stream = riva.client.audio_io.SoundCallBack(
                args.output_device, nchannels=nchannels, sampwidth=sampwidth, framerate=args.sample_rate_hz
            )
        if args.output is not None:
            wav_out = wave.open(args.output, 'wb')
            wav_out.setnchannels(nchannels)
            wav_out.setsampwidth(sampwidth)
            wav_out.setframerate(args.sample_rate_hz)

        with riva.client.audio_io.MicrophoneStream(
            args.sample_rate_hz,
            args.file_streaming_chunk,
            device=args.input_device,
        ) as mic_stream:
            asr_responses = asr_service.streaming_response_generator(
                audio_chunks=mic_stream,
                streaming_config=asr_config,
            )
                
            #riva.client.print_streaming(responses=asr_responses, additional_info='confidence') #show_intermediate=True, 
            transcript = ''
            
            for asr_response in asr_responses:
                if not asr_response.results:
                    continue
                #print(asr_response)
                for asr_result in asr_response.results:
                    new_transcript = asr_result.alternatives[0].transcript
                    if not asr_result.is_final:
                        if transcript != new_transcript:  # only print updates
                            transcript = new_transcript
                            print('>>', transcript)
                        continue
                    transcript = new_transcript
                    print('##', transcript)
                    tts_responses = tts_service.synthesize_online(
                        transcript, args.voice, args.language_code, sample_rate_hz=args.sample_rate_hz
                    )
                    for tts_response in tts_responses:
                        if speaker_stream is not None:
                            speaker_stream(tts_response.audio)
                        if wav_out is not None:
                            wav_out.writeframesraw(tts_response.audio)
    finally:
        if wav_out is not None:
            wav_out.close()
        if speaker_stream is not None:
            speaker_stream.close()


if __name__ == '__main__':
    main()
    