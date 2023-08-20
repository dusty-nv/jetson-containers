// https://web.dev/patterns/media/microphone-process/
// https://gist.github.com/flpvsk/047140b31c968001dc563998f7440cc1
// https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor
// https://stackoverflow.com/questions/57583266/audioworklet-set-output-to-float32array-to-stream-live-audio

class AudioCaptureProcessor extends AudioWorkletProcessor {
  process([inputs], [outputs], parameters) {
		//console.log(`AudioCaptureProcessor::process(${inputs.length}, ${outputs.length})`);
    //console.log(inputs);
		
		// convert float->int16 samples
		var input = inputs[0]; 
		var samples = new Int16Array(input.length);
		
		for( let i=0; i < input.length; i++ )
			samples[i] = 32767 * Math.min(Math.max(input[i], -1), 1);
		
		this.port.postMessage({
			'type': 'audio',
			'data': samples
		});
		
		// relay outputs (remove connect(audioContext.destination) to disable)
		for( let i=0; i < inputs.length && i < outputs.length; i++ )
			outputs[i].set(inputs[i]);
		
    return true;
  }
}

registerProcessor("AudioCaptureProcessor", AudioCaptureProcessor);