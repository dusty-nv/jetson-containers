/*
 * handles audio input/output device streaming
 * websocket.js should be included before this
 */

var audioContext;       // AudioContext
var audioInputTrack;    // MediaStreamTrack
var audioInputDevice;   // MediaStream
var audioInputStream;   // MediaStreamAudioSourceNode
var audioInputCapture;  // AudioWorkletNode
var audioOutputWorker;  // AudioWorkletNode

//var audioMicRecorder;


function checkMediaDevices() {
  return (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !navigator.mediaDevices.enumerateDevices) ? false : true;
}

function enumerateAudioDevices() {
	var selectInput = document.getElementById('audio-input-select');
	var selectOutput = document.getElementById('audio-output-select');
	
	if( !checkMediaDevices() ) {
		selectInput.add(new Option('use HTTPS to enable browser audio'));
		selectOutput.add(new Option('use HTTPS to enable browser audio'));
		return;
	}
	
	navigator.mediaDevices.getUserMedia({audio: true, video: false}).then((stream) => { // get permission from user
		navigator.mediaDevices.enumerateDevices().then((devices) => {
			stream.getTracks().forEach(track => track.stop()); // close the device opened to get permissions
			devices.forEach((device) => {
				console.log(`Browser media device:  ${device.kind}  label=${device.label}  id=${device.deviceId}`);
				
				if( device.kind == 'audioinput' )
					selectInput.add(new Option(device.label, device.deviceId));
				else if( device.kind == 'audiooutput' )
					selectOutput.add(new Option(device.label, device.deviceId));
			});
			
			if( selectInput.options.length == 0 )
				selectInput.add(new Option('browser has no audio inputs available'));

			if( selectOutput.options.length == 0 )
				selectOutput.add(new Option('browser has no audio outputs available'));
		});
	}).catch(reportError);
}

function openAudioDevices(inputDeviceId, outputDeviceId) {
	if( inputDeviceId == undefined )
		inputDeviceId = document.getElementById('audio-input-select').value;
	
	if( outputDeviceId == undefined )
		outputDeviceId = document.getElementById('audio-output-select').value;
	
	const constraints = {
		video: false,
		audio: {
			deviceId: inputDeviceId
		},
	};
	
	navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
		console.log('Opened audio input device %s', inputDeviceId);

    audioInputDevice = stream;
		audioInputTrack = stream.getAudioTracks()[0];
	  audioSettings = audioInputTrack.getSettings();
		
		console.log(audioInputTrack);
		console.log(audioSettings);
		
		/*options = {
			//mimeType: 'audio/webm; codecs=opus',
			mimeType: 'audio/webm; codecs=pcm',
		}
		
		audioMicRecorder = new MediaRecorder(audioInputDevice, options);
		audioMicRecorder.ondataavailable = onMicAudio;
		audioMicRecorder.start(250);  // capture interval in milliseconds */
		
		options = {
			//'latencyHint': 1.0,
			'sampleRate': audioSettings.sampleRate,
			'sinkId': outputDeviceId,
		};
		
		audioContext = new AudioContext(options);
		audioInputStream = audioContext.createMediaStreamSource(audioInputDevice);
		audioContext.audioWorklet.addModule("/static/audioWorkers.js").then(() => {
			audioInputCapture = new AudioWorkletNode(audioContext, "AudioCaptureProcessor");
			audioOutputWorker = new AudioWorkletNode(audioContext, "AudioOutputProcessor");
			audioInputStream.connect(audioInputCapture).connect(audioOutputWorker).connect(audioContext.destination);
			audioInputCapture.port.onmessage = onAudioInputCapture;
		});
	}).catch(reportError);
}

function onAudioInputCapture(event) {
	msg = event.data;
	
	if( msg['type'] != 'audio' )
		return;

	// encode to base64
	var reader = new FileReader();
	reader.readAsDataURL(new Blob([msg['data']]));
	
	reader.onloadend = function () {
		json = JSON.stringify({
				'type': 'audio',
				'size': msg['data'].length * 2,  // 16-bit samples
				'data': reader.result.slice(reader.result.indexOf(',') + 1), // remove the `data:...;base64,` header
				'settings': audioInputTrack.getSettings(),
			});
		websocket.send(json);
	};
}

function onAudioOutput(samples) {
	if( audioOutputWorker != undefined ) {
		int16Array = new Int16Array(samples);
		audioOutputWorker.port.postMessage(int16Array, [int16Array.buffer]);
	}
}

/*function onMicAudio(event) {  // previous handler used with MediaRecorder
	data = event.data;
	console.log(`onMicAudio()  size=${data.size} type=${data.type}`);
	
	// encode to base64
	var reader = new FileReader();
	reader.readAsDataURL(data);
	
	reader.onloadend = function () {
		json = JSON.stringify({
				'type': 'audio',
				'size': data.size,
				'mime': data.type,
				'data': reader.result.slice(reader.result.indexOf(',') + 1), // remove the `data:...;base64,` header
			});
		websocket.send(json);
	};
}*/
