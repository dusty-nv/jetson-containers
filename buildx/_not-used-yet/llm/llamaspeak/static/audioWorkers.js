// https://web.dev/patterns/media/microphone-process/
// https://gist.github.com/flpvsk/047140b31c968001dc563998f7440cc1
// https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor
// https://stackoverflow.com/questions/57583266/audioworklet-set-output-to-float32array-to-stream-live-audio
// https://github.com/GoogleChromeLabs/web-audio-samples/tree/main/src/audio-worklet/design-pattern/wasm-ring-buffer


class AudioCaptureProcessor extends AudioWorkletProcessor {
  process([inputs], [outputs], parameters) {
		//console.log(`AudioCaptureProcessor::process(${inputs.length}, ${outputs.length})`);
		// convert float->int16 samples
		var input = inputs[0]; 
		var samples = new Int16Array(input.length);

		for( let i=0; i < input.length; i++ )
			samples[i] = 32767 * Math.min(Math.max(input[i], -1), 1);
		
		this.port.postMessage(samples, [samples.buffer]);
		
		// relay outputs
		//for( let i=0; i < inputs.length && i < outputs.length; i++ )
		//	outputs[i].set(inputs[i]);
		
    return true;
  }
}

class AudioOutputProcessor extends AudioWorkletProcessor {
	constructor(options) {
		super();

		this.queue = [];
		this.playhead = 0;

		this.port.onmessage = this.onmessage.bind(this);
	}
	
	onmessage(event) {
		const { data } = event;
		this.queue.push(data);
	}
	
	process([inputs], [outputs], parameters) {
		const output = outputs[0];
		var samplesWritten = 0;
		
		/*for(let i = 0; i < output.length; i++) {
				output[i] = Math.sin(this.sampleCount * (Math.sin(this.sampleCount/24000.0) + 440.0) * Math.PI * 2.0 / 48000.0); //* 32767;
				this.sampleCount++;
		}*/
		
		//console.log(`audio queue length ${this.queue.length} ${output.length} ${this.playhead}`);
		
		while( this.queue.length > 0 && samplesWritten < output.length ) {
			for( let i=samplesWritten; i < output.length && this.playhead < this.queue[0].length; i++ ) {
				output[i] = this.queue[0][this.playhead] / 32767.0;
				this.playhead++;
				samplesWritten++;
			}
			
			if( this.playhead >= this.queue[0].length ) {
				this.queue.shift();
				this.playhead = 0;
			}
		}

		/*if( samplesWritten < output.length ) {
			console.warn(`gap in output audio  (${samplesWritten} of ${output.length} samples written)`);
		}*/
		
		for( let i=samplesWritten; i < output.length; i++ ) {
			output[i] = 0;
		}
		
		for( let i=1; i < outputs.length; i++ )
			outputs[i].set(outputs[0]);
		
    return true;
  }
}

registerProcessor("AudioCaptureProcessor", AudioCaptureProcessor);
registerProcessor("AudioOutputProcessor", AudioOutputProcessor);
