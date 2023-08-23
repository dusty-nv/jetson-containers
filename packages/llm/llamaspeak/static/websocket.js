/*
 * handles all the websocket streaming of text and audio to/from the server
 */
var websocket;
var last_msg_id;

function reportError(msg) {
  console.log(msg);
}
 
function getWebsocketProtocol() {
  return window.location.protocol == 'https:' ? 'wss://' : 'ws://';
}

function getWebsocketURL(port=49000) {  // wss://192.168.1.2:49000
  return `${getWebsocketProtocol()}${window.location.hostname}:${port}/${name}`;
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
		//console.log(`recieved websocket message:  id=${msg_id}  type=${msg_type}  timestamp=${timestamp}  latency=${latency}  magic_number=${magic_number}  payload_size=${payload_size}`);
		
		if( magic_number != 42 ) {
			console.log(`recieved invalid websocket msg (magic_number=${magic_number}  size=${msg.size}`);
		}
		
		if( payload_size != payload.size ) {
			console.log(`recieved invalid websocket msg (payload_size=${payload_size} actual=${payload.size}`);
		}
		
		if( last_msg_id != undefined && msg_id != (last_msg_id + 1) )
			console.log(`warning:  out-of-order message ID ${msg_id}  (last=${last_msg_id})`);
			
		last_msg_id = msg_id;
		
		if( msg_type == 1 ) {
			payload.text().then((text) => {
				console.log(`text message: ${text}`);
			});
		}
		else if( msg_type == 2 ) {
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
