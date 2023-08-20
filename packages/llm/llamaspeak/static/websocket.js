/*
 * handles all the websocket streaming of text and audio to/from the server
 */
var websocket;

var audioContext;      // AudioContext

var audioMicTrack;     // MediaStreamTrack
var audioMicDevice;    // MediaStream
var audioMicStream;    // MediaStreamAudioSourceNode
var audioMicCapture;   // AudioWorkletNode

//var audioMicRecorder;


function reportError(msg) {
  console.log(msg);
}
 
function getWebsocketProtocol() {
  return window.location.protocol == 'https:' ? 'wss://' : 'ws://';
}

function getWebsocketURL(port=49000) {  // wss://192.168.1.2:49000
  return `${getWebsocketProtocol()}${window.location.hostname}:${port}/${name}`;
}
  
function checkMediaDevices() {
  return (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !navigator.mediaDevices.enumerateDevices) ? false : true;
}

function getUserMessageStyle(user) {
	var color = '#999999';
	
	switch(user) {
		case 0: color = '#659864'; break; //'#347B98' '5F8D4E'
		case 1: color = '#56718F'; break; //'#559E54' 
  }

	return `background-color: ${color}; border-color: ${color};`
}

function onWebsocket(event) {
	var msg;
  var url = event.srcElement.url;

  try {
    msg = JSON.parse(event.data);
  } catch (e) {
    return;
  }

	if( msg['type'] == 'message' ) {
		prev_msg = $(`#container #message_${msg['id']}`);
		if( prev_msg.length > 0 ) {
			prev_msg.html(msg['text']);
	  }
		else {
			$('#container').append(`<br/><span id="message_${msg['id']}" class="message_body" style="${getUserMessageStyle(msg['user'])}">`
									+ msg['text'] +
									'</span><br/><br/>');
		}
	}
	
  //console.log('Websocket message recieved:');
	//console.log(msg);
}

function connectWebsocket() {
	websocket = new WebSocket(getWebsocketURL());
	websocket.addEventListener('message', onWebsocket);
}

function setupAudio() {
	var selectMic = document.getElementById('audio-mic-select');
	
	if( !checkMediaDevices() ) {
		selectMic.add(new Option('use HTTPS to enable browser audio'));
		//sendButton.disabled = true;
		return;
	}
	
	navigator.mediaDevices.getUserMedia({audio: true, video: false}).then((stream) => { // get permission from user
		navigator.mediaDevices.enumerateDevices().then((devices) => {
			stream.getTracks().forEach(track => track.stop()); // close the device opened to get permissions
			devices.forEach((device) => {
				if( device.kind == 'audioinput' ) {
					console.log(`Browser media device:  ${device.kind}  label=${device.label}  id=${device.deviceId}`);
					selectMic.add(new Option(device.label, device.deviceId));
				}
			});
			if( selectMic.options.length == 0 ) {
				selectMic.add(new Option('browser has no audio inputs available'));
				//sendButton.disabled = true;
			}
			else {
				onMicSelect(); // automatically send default mic
			}
		});
	}).catch(reportError);
}

function sendAudio(deviceId) {
	const constraints = {
		video: false,
		audio: {
			deviceId: deviceId
		},
	};
	
	navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
		console.log('Adding audio stream (deviceId=%s)', deviceId);

    audioMicDevice = stream;
		audioMicTrack = stream.getAudioTracks()[0];
	  audioSettings = audioMicTrack.getSettings();
		
		console.log(audioMicTrack);
		console.log(audioSettings);
		
		/*options = {
			//mimeType: 'audio/webm; codecs=opus',
			mimeType: 'audio/webm; codecs=pcm',
		}
		
		audioMicRecorder = new MediaRecorder(audioMicDevice, options);
		audioMicRecorder.ondataavailable = onMicAudio;
		audioMicRecorder.start(250);  // capture interval in milliseconds */
		
		options = {
			//'latencyHint': 1.0,
			'sampleRate': audioSettings.sampleRate,
		};
		
		audioContext = new AudioContext(options);
		audioMicStream = audioContext.createMediaStreamSource(audioMicDevice);
		audioContext.audioWorklet.addModule("/static/audio.js").then(() => {
			audioMicCapture = new AudioWorkletNode(audioContext, "AudioCaptureProcessor");
			audioMicStream.connect(audioMicCapture); //.connect(audioContext.destination);
			audioMicCapture.port.onmessage = onMicAudio;
		});
	}).catch(reportError);
}

function onMicAudio(event) {
	//console.log('onMicAudio', event.data);
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
				'settings': audioMicTrack.getSettings(),
			});
		websocket.send(json);
	};
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

