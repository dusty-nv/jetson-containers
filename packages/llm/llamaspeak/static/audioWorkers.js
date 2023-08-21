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
		
		this.port.postMessage({
			'type': 'audio',
			'data': samples
		});
		
		// relay outputs
		//for( let i=0; i < inputs.length && i < outputs.length; i++ )
		//	outputs[i].set(inputs[i]);
		
    return true;
  }
}

class AudioOutputProcessor extends AudioWorkletProcessor {
	constructor(options) {
		super();
		
		this.bufferLength = options.processorOptions.bufferLength;
    this.channelCount = options.processorOptions.channelCount;
		
		this.ringBuffer = new RingBuffer(this.bufferLength, this.channelCount);
		
		this.port.onmessage = this.onmessage.bind(this);
	}
	
	onmessage(event) {
		const { data } = event;

		let int16Array = new Int16Array(data);
		let floatArray = new Float32Array(int16Array.length);

		for( let i=0; i < int16Array.length; i++ )
			floatArray[i] = int16Array[i] / 32767.0;

		this.ringBuffer.push([floatArray]);
	}
	
	process([inputs], [outputs], parameters) {
		//console.log(`AudioOutputProcessor::process(${inputs.length}, ${outputs.length})`);

		/*const output = outputs[0];
		output.forEach((channel) => {
      for (let i = 0; i < channel.length; i++) {
        channel[i] = Math.random() * 2 - 1;
      }
    });*/
		
		// pull 128 frames out (output will be silent if not enough frames in buffer)
		this.ringBuffer.pull([outputs[0]]);
		
		for( let i=1; i < outputs.length; i++ )
			outputs[i].set(outputs[0]);
		
    return true;
  }
}

registerProcessor("AudioCaptureProcessor", AudioCaptureProcessor);
registerProcessor("AudioOutputProcessor", AudioOutputProcessor);


/**
 * A JS FIFO implementation for the AudioWorklet. 3 assumptions for the
 * simpler operation:
 *  1. the push and the pull operation are done by 128 frames. (Web Audio
 *    API's render quantum size in the speficiation)
 *  2. the channel count of input/output cannot be changed dynamically.
 *    The AudioWorkletNode should be configured with the `.channelCount = k`
 *    (where k is the channel count you want) and
 *    `.channelCountMode = explicit`.
 *  3. This is for the single-thread operation. (obviously)
 *
 * https://github.com/GoogleChromeLabs/web-audio-samples/blob/main/src/audio-worklet/design-pattern/lib/wasm-audio-helper.js
 */
class RingBuffer {
  /**
   * @constructor
   * @param  {number} length Buffer length in frames.
   * @param  {number} channelCount Buffer channel count.
   */
  constructor(length, channelCount) {
    this._readIndex = 0;
    this._writeIndex = 0;
    this._framesAvailable = 0;

    this._channelCount = channelCount;
    this._length = length;
    this._channelData = [];
    for (let i = 0; i < this._channelCount; ++i) {
      this._channelData[i] = new Float32Array(length);
    }
  }

  /**
   * Getter for Available frames in buffer.
   *
   * @return {number} Available frames in buffer.
   */
  get framesAvailable() {
    return this._framesAvailable;
  }

  /**
   * Push a sequence of Float32Arrays to buffer.
   *
   * @param  {array} arraySequence A sequence of Float32Arrays.
   */
  push(arraySequence) {
    // The channel count of arraySequence and the length of each channel must
    // match with this buffer obejct.

    // Transfer data from the |arraySequence| storage to the internal buffer.
    const sourceLength = arraySequence[0].length;
    for (let i = 0; i < sourceLength; ++i) {
      const writeIndex = (this._writeIndex + i) % this._length;
      for (let channel = 0; channel < this._channelCount; ++channel) {
        this._channelData[channel][writeIndex] = arraySequence[channel][i];
      }
    }

    this._writeIndex += sourceLength;
    if (this._writeIndex >= this._length) {
      this._writeIndex = 0;
    }

    // For excessive frames, the buffer will be overwritten.
    this._framesAvailable += sourceLength;
    if (this._framesAvailable > this._length) {
      this._framesAvailable = this._length;
    }
  }

  /**
   * Pull data out of buffer and fill a given sequence of Float32Arrays.
   *
   * @param  {array} arraySequence An array of Float32Arrays.
   */
  pull(arraySequence) {
    // The channel count of arraySequence and the length of each channel must
    // match with this buffer obejct.

    // If the FIFO is completely empty, do nothing.
    if (this._framesAvailable === 0) {
      return;
    }

    const destinationLength = arraySequence[0].length;

    // Transfer data from the internal buffer to the |arraySequence| storage.
    for (let i = 0; i < destinationLength; ++i) {
      const readIndex = (this._readIndex + i) % this._length;
      for (let channel = 0; channel < this._channelCount; ++channel) {
        arraySequence[channel][i] = this._channelData[channel][readIndex];
      }
    }

    this._readIndex += destinationLength;
    if (this._readIndex >= this._length) {
      this._readIndex = 0;
    }

    this._framesAvailable -= destinationLength;
    if (this._framesAvailable < 0) {
      this._framesAvailable = 0;
    }
  }
}
