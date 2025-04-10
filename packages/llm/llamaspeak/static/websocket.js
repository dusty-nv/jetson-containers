/*
 * handles all the websocket streaming of text and audio to/from the server
 */
var websocket;

var msg_count_rx;
var msg_count_tx=0;

function reportError(msg) {
  console.log(msg);
}
 
function getWebsocketProtocol() {
  return window.location.protocol == 'https:' ? 'wss://' : 'ws://';
}

function getWebsocketURL(port=49000) {  // wss://192.168.1.2:49000
  return `${getWebsocketProtocol()}${window.location.hostname}:${port}/${name}`;
}

function sendWebsocket(payload, type=0) {
  const timestamp = Date.now();	
	let header = new DataView(new ArrayBuffer(32));
		
	header.setBigUint64(0, BigInt(msg_count_tx));
	header.setBigUint64(8, BigInt(timestamp));
	header.setUint16(16, 42);
	header.setUint16(18, type);
	
	msg_count_tx++;
	
	let payloadSize;
	
	if( payload instanceof ArrayBuffer || ArrayBuffer.isView(payload) ) { // binary
		payloadSize = payload.byteLength;
	}
	else if( payload instanceof Blob) {
		payloadSize = payload.size;
	}
	else { // serialize to JSON
		payload = new TextEncoder().encode(JSON.stringify(payload)); // Uint8Array
		payloadSize = payload.byteLength;
	}
	
	header.setUint32(20, payloadSize);
	
	//console.log(`sending ${typeof payload} websocket message (type=${type} timestamp=${timestamp} payload_size=${payloadSize})`);
	websocket.send(new Blob([header, payload]));
}

function onWebsocket(event) {

	//console.log('recieved websocket msg', event);
	const msg = event.data;
	
	if( msg.size <= 32 ) {
		console.log(`recieved invalid websocket msg (size=${msg.size})`);
		return;
	}
	
	const header = msg.slice(0, 32);
	const payload = msg.slice(32);
	
	header.arrayBuffer().then((headerBuffer) => {
		const view = new DataView(headerBuffer);
		
		const msg_id = Number(view.getBigUint64(0));
		const timestamp = view.getBigUint64(8);
		const magic_number = view.getUint16(16);
		const msg_type = view.getUint16(18);
		const payload_size = view.getUint32(20);
		
		//const latency = BigInt(Date.now()) - timestamp;  // this is negative?  sub-second client/server time sync needed
		//console.log(`recieved websocket message:  id=${msg_id}  type=${msg_type}  timestamp=${timestamp}  latency=${latency}  payload_size=${payload_size}`);
		
		if( magic_number != 42 ) {
			console.log(`recieved invalid websocket msg (magic_number=${magic_number}  size=${msg.size}`);
		}
		
		if( payload_size != payload.size ) {
			console.log(`recieved invalid websocket msg (payload_size=${payload_size} actual=${payload.size}`);
		}
		
		if( msg_count_rx != undefined && msg_id != (msg_count_rx + 1) )
			console.log(`warning:  out-of-order message ID ${msg_id}  (last=${msg_count_rx})`);
			
		msg_count_rx = msg_id;
		
		if( msg_type == 0 ) { // JSON message
			payload.text().then((text) => {
				json = JSON.parse(text);
				//console.log('json message:', json);
				
				if( 'chat_history' in json ) {
					const chat_history = json['chat_history'];
					
					var chc = document.getElementById('chat-history-container');
					var isScrolledToBottom = chc.scrollHeight - chc.clientHeight <= chc.scrollTop + 1;
					
					$('#chat-history-container').empty(); // started clearing because server may remove partial/rejected ASR prompts
					
					for( let n=0; n < chat_history.length; n++ ) {
						for( let m=0; m < chat_history[n].length; m++ ) {
							prev_msg = $(`#chat-history-container #msg_${n}_${m}`);
							if( prev_msg.length > 0 ) {
								prev_msg.html(chat_history[n][m]);
							}
							else {
								$('#chat-history-container').append(
									`<div id="msg_${n}_${m}" class="chat-message-user-${m} mb-3">${chat_history[n][m]}</div><br/>`
								);
							}
						}
					}
					
					if( isScrolledToBottom ) // autoscroll unless the user has scrolled up
						chc.scrollTop = chc.scrollHeight - chc.clientHeight;
				}
				
				if( 'tegrastats' in json ) {
					console.log(json['tegrastats']);
				}
			});
		}
		if( msg_type == 1 ) { // TEXT message
			payload.text().then((text) => {
				console.log(`text message: ${text}`);
			});
		}
		else if( msg_type == 2 ) { // AUDIO message
			payload.arrayBuffer().then((payloadBuffer) => {
				onAudioOutput(payloadBuffer);
			});
		}
	});
	
  /*try {
    msg = JSON.parse(event.data);
  } catch (e) {
    return;
  }

	if( msg['type'] == 'message' ) {
		prev_msg = $(`#chat-history-container #message_${msg['id']}`);
		if( prev_msg.length > 0 ) {
			prev_msg.html(msg['text']);
	  }
		else {
			$('#chat-history-container').append(`<br/><span id="message_${msg['id']}" class="chat-message-body" style="${getUserMessageStyle(msg['user'])}">`
									+ msg['text'] +
									'</span><br/><br/>');
		}
	}
	else if( msg['type'] == 'audio' ) {
		//console.log('RECIEVED AUDIO MSG', msg['data'].length, msg);
		//onAudioOutput(msg['data']);
	}*/
	
	
	
  //console.log('Websocket message recieved:');
	//console.log(msg);
}

function connectWebsocket() {
	websocket = new WebSocket(getWebsocketURL());
	websocket.addEventListener('message', onWebsocket);
}
