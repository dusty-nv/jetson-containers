/*
 * handles all the websocket streaming of text and audio to/from the server
 */
var websocket;

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
	else if( msg['type'] == 'audio' ) {
		//console.log('RECIEVED AUDIO MSG', msg);
		onAudioOutput(msg['data']);
	}
	
  //console.log('Websocket message recieved:');
	//console.log(msg);
}

function connectWebsocket() {
	websocket = new WebSocket(getWebsocketURL());
	websocket.addEventListener('message', onWebsocket);
}
