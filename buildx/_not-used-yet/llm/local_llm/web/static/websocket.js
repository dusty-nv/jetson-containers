/*
 * Handles all the websocket streaming of text and audio to/from the server.
 * This supports sending and recieving of JSON, text, and binary messages.
 * It's a simple interface and only uses on connection to the server:
 *
 *   - First, call connectWebsocket() and provide your message handler function.
 *     This message handler will be called when messages are recieved from the server.
 *
 *   - Message callbacks should be of the signature `on_message(payload, type)`,
 *     where type is MESSAGE_JSON, MESSAGE_TEXT, or MESSAGE_BINARY.
 *
 *        > The payload for JSON messages will be a dict.
 *        > The payload for text messages will be a string.
 *        > The payload for binary messages will be an ArrayBuffer.
 *
 *   - To send messages to the server, use the sendWebsocket() function
 *     and provide the payload and one of the supported message types. 
 */
var websocket;
var websocketCallback;

var msg_count_rx;
var msg_count_tx=0;

const MESSAGE_JSON = 0;
const MESSAGE_TEXT = 1;
const MESSAGE_BINARY = 2;
const MESSAGE_FILE = 3;
const MESSAGE_AUDIO = 4;
const MESSAGE_IMAGE = 5;

function reportError(msg) {
  console.log(msg);
}
 
function getWebsocketProtocol() {
  return window.location.protocol == 'https:' ? 'wss://' : 'ws://';
}

function connectWebsocket(msgCallback, port=49000) {
	websocketCallback = msgCallback;
	websocket = new WebSocket(`${getWebsocketProtocol()}${window.location.hostname}:${port}`);
	websocket.addEventListener('message', websocketListener);
}

function sendWebsocket(payload, type=MESSAGE_JSON, metadata='') {
  const timestamp = Date.now();	
	let header = new DataView(new ArrayBuffer(24));
		
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

	var metadata_buffer = new Uint8Array(8);
	
	if( metadata ) {
		const metadata_utf8 = new TextEncoder().encode(metadata);
		for( let i=0; i < metadata_buffer.length && i < metadata_utf8.length; i++ )
			metadata_buffer[i] = metadata_utf8[i];
	}
	
	//console.log(`sending ${typeof payload} websocket message (type=${type} timestamp=${timestamp} payload_size=${payloadSize})`);
	websocket.send(new Blob([header, metadata_buffer, payload]));
}

function websocketUpload(dataTransfer) {
	if (dataTransfer.items) {
		// Use DataTransferItemList interface to access the file(s)
		[...dataTransfer.items].forEach((item, i) => { 
			if( item.kind != "file" ) // If dropped items aren't files, reject them
				return;

			const file = item.getAsFile();
			const ext = file.name.split('.').pop().toLowerCase();
			const img_exts = ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif', 'webm'];
			
			console.log(`DataTransferItemList file[${i}].name=${file.name}  ext=${ext}  mime=${file.type}`);

			if( !img_exts.some(img_ext => ext == img_ext) )
				return;

			file.arrayBuffer().then((fileBuffer) => {
				sendWebsocket(fileBuffer, type=MESSAGE_IMAGE, metadata=ext);
			});
			
		});
	} else {
		// Use DataTransfer interface to access the file(s)
		[...dataTransfer.files].forEach((file, i) => {
			console.log(`DataTransfer file[${i}].name = ${file.name}`);
			console.error('Browser does not support DataTransferItemList');
		});
	}
}

function websocketListener(event) {
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
		
		if( msg_type == MESSAGE_JSON ) { 
			payload.text().then((text) => {
				json = JSON.parse(text);
				console.log('recieved json websocket message:', json);
				
				if( websocketCallback != undefined )
						websocketCallback(json, msg_type);
			});
		}
		if( msg_type == MESSAGE_TEXT ) { // TEXT message
			payload.text().then((text) => {
				console.log(`recieved text websocket message: ${text}`);
				if( websocketCallback != undefined )
					websocketCallback(text, msg_type);
			});
		}
		else if( msg_type >= MESSAGE_BINARY ) { 
			payload.arrayBuffer().then((payloadBuffer) => {
				if( websocketCallback != undefined )
					websocketCallback(payloadBuffer, msg_type);
			});
		}
	});
}