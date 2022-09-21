// Copyright Jacob C. Stucki III
 'use strict';
var express = require('express');
var app = express();
var server = require('http').createServer(app);
var io = require('socket.io')(server); //server mounts on node.js http server

//Listen to port on IP
server.listen(8080, function(){
	console.log('Listening on *:8080');
})


//Messages from clients
io.on('connection', function(client) {

	//On Connect
	console.log('Client connected')
	// console.log("Current Connections: ", clients_connected)
	// // console.log(client.id)

	// //On Disconnect
	// client.on('disconnect', function(){
	// 	console.log('Current Connections: ', clients_connected)
	// });

	client.on('dictionary', function(data){
		console.log('dict recieved')
		io.emit('dictionary',data)
	})

});





