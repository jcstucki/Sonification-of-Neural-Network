'use strict';
//Copyright 2021 Jacob C. Stucki III


// Max interaction setup
var maxapi = require('max-api')
maxapi.post('running') //post prints to max console

//args setup
var args = process.argv.slice(2) // removes 'node' and 'script.js' from arguments
var server = ''
var port = ''
args.forEach(function(val, index, array){
	if (val == '-webserver'){ 
		server = args[index+1]
		maxapi.post('Server: '+server)
	}
	if (val == '-port'){ 
		port = args[index+1]
		maxapi.post('Port: '+port)
	}
})





//socket.io client to server connection
var server_port_string = 'http://'+server + ':' + port;
maxapi.post(server_port_string)

var io =  require('socket.io-client');
var socket = io.connect(server_port_string,{reconnect: true});

socket.on('connect', function(){
	maxapi.post('Socket.io Connected')
})

socket.on('disconnect', function(){
	maxapi.post('Socket.io Disconnected, attempting auto-reconnect')
})

socket.on('dictionary',function(data){
	//maxapi.post('Recieved dictionary')
	maxapi.outlet(data)
})

//Read from stdin for messages from User
// var readline = require('readline');
// var rl = readline.createInterface({
// 	input:process.stdin,
// 	output:process.stdout,
// 	terminal:false
// });

// rl.on('line', function(line){
// 	if (line == 'reset'){
// 		socket.emit('reset');
// 		maxapi.post('Sent Reset Message to Server');
// 	};
// });

// socket.on('recieved_reset',function(){
// 	maxapi.post('Server Reset Sliders')
// })

