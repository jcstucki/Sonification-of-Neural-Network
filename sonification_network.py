from objects import *

from p5 import *
import numpy as np
import math
import random
import time
import socketio
import pandas as pd


sio = socketio.Client()
sio.connect('http://localhost:8080')
print('SID:', sio.sid)


#Load data https://amentum.io/spaceradiation_docs#operation/app.api.endpoints.TrappedRadiation.calculate_flux_percentile

data_df = pd.read_csv('2019_all_months_all_days')
print(data_df)
data_df['flux_normalized'] = data_df['flux'] / max(data_df['flux'])

t = 0

#Inputs
input_nodes = []
input_1 = Neuron(y=400, x = 100, size = 6)
input_2 = Neuron(y= 500 , x = 100, size = 6)
input_3 = Neuron(y = 600, x = 100, size = 6)
input_nodes.append(input_1)
input_nodes.append(input_2)
input_nodes.append(input_3)

#Hidden
num_hidden = 3
layer_1_nodes = []
for i in range(1, num_hidden+1):
	x = 300
	y = 100 * i + 100
	layer_1_nodes.append(Neuron(x = x, y = y, size = 6))

layer_2_nodes = []
for i in range(1, num_hidden+1):
	x = 400
	y = 100 * i + 100
	layer_2_nodes.append(Neuron(x = x, y = y, size = 6))

layer_3_nodes = []
for i in range(1, num_hidden+1):
	x = 500
	y = 100 * i + 100
	layer_3_nodes.append(Neuron(x = x, y = y, size = 6))

#Outputs
output_nodes = []
output_1 = Neuron(y = 400, x = 850, size = 6)
output_nodes.append(output_1)
# output_2 = Neuron(y = 500, x = 850, size = 6)
# output_nodes.append(output_2)
# output_3 = Neuron(y = 600, x = 850, size = 6)
# output_nodes.append(output_3)



#Training n
training_nodes = []
training_1 = Neuron(x = 950, y= 400, size = 6)
training_nodes.append(training_1)
# training_2 = Neuron(x = 950, y = 500, size = 6)
# training_nodes.append(training_2)
# training_3 = Neuron(x = 950, y = 600, size = 6)
# training_nodes.append(training_3)


#Connect Inputs 
for x1 in input_nodes:
	for x2 in layer_1_nodes:
		x2.input_connections.append(Connection(input_ = x1, output = x2)) #put inputs in output connection so we can use for backpropogation

for x1 in layer_1_nodes:
	for x2 in layer_2_nodes:
		x2.input_connections.append(Connection(input_ = x1, output = x2))

for x1 in layer_2_nodes:
	for x2 in layer_3_nodes:
		x2.input_connections.append(Connection(input_ = x1, output = x2))

for x1 in layer_3_nodes:
	for x2 in output_nodes:
		x2.input_connections.append(Connection(input_ = x1, output = x2))


#Skip Connections

#inputs to layers
# for x1 in input_nodes:
# 	for x2 in layer_2_nodes:
# 		x2.input_connections.append(Connection(input_ = x1, output = x2))

# for x1 in input_nodes:
# 	for x2 in layer_3_nodes:
# 		x2.input_connections.append(Connection(input_ = x1, output = x2))

# # #layers to layers
# for x1 in layer_1_nodes:
# 	for x2 in layer_3_nodes:
# 		x2.input_connections.append(Connection(input_ = x1, output = x2))


# #layers to output
# for x1 in layer_1_nodes:
# 	for x2 in output_nodes:
# 		x2.input_connections.append(Connection(input_ = x1, output = x2))
# for x1 in layer_2_nodes:
# 	for x2 in output_nodes:
# 		x2.input_connections.append(Connection(input_ = x1, output = x2))


#output to training labels

c1 = Connection(input_ = output_1, output = training_1)
c1.weight = 1
training_1.input_connections.append(c1)

# c2 = Connection(input_ = output_2, output = training_2)
# c2.weight = 1
# training_2.input_connections.append(c2)

# c3 = Connection(input_ = output_3, output = training_3)
# c3.weight = 1
# training_3.input_connections.append(c3)


def setup():
	size(1000,1000)
	background(255)

	#Pre Train
	offset = 10
	epochs = 2000
	for e in range(epochs):
		print(e)
		for i in range(offset, len(data_df)-1):
			# print(i)
			input_1.output = data_df['flux_normalized'].values[i]
			input_2.output = data_df['flux_normalized'].values[i-1]
			input_3.output = data_df['flux_normalized'].values[i-2]
			training_1.output = data_df['flux_normalized'].values[i+1]

			#Forward
			for n in layer_1_nodes:
				n.weight_sum = 0
				for c in n.input_connections:
					n.sum_inputs(input_ = c.input.output, weight = c.weight)
				n.activate()

			for n in layer_2_nodes:
				n.weight_sum = 0
				for c in n.input_connections:
					n.sum_inputs(input_ = c.input.output, weight = c.weight)
				n.activate()

			for n in layer_3_nodes:
				n.weight_sum = 0
				for c in n.input_connections:
					n.sum_inputs(input_ = c.input.output, weight = c.weight)
				n.activate()

			for n in output_nodes:
				n.weight_sum = 0
				for c in n.input_connections:
					n.sum_inputs(input_ = c.input.output, weight = c.weight)
				n.activate()


			#Backpropogate Error for training data
			sigmoid_dir = lambda x: x * (1 - x)
			gcu_dir = lambda x: math.cos(x) - x * math.sin(x)
		
			#Output Layer Error #NOTE, DIfferent errors for output and hidden
			for training_n in training_nodes: #output neuron
				for c in training_n.input_connections:
					input_n = c.input #input neuron
					input_n.error += (input_n.output - training_n.output) * sigmoid_dir(input_n.output)
					
			# Hidden Layer Error
			for output_n in output_nodes:
				for c in output_n.input_connections:
					input_n = c.input
					input_n.error += c.weight * output_n.error * sigmoid_dir(output_n.output)

			for output_n in layer_3_nodes:
				for c in output_n.input_connections:
					input_n = c.input
					input_n.error += c.weight * output_n.error * sigmoid_dir(output_n.output)

			for output_n in layer_2_nodes:
				for c in output_n.input_connections:
					input_n = c.input
					input_n.error += c.weight * output_n.error * sigmoid_dir(output_n.output)

			for output_n in layer_1_nodes:
				for c in output_n.input_connections:
					input_n = c.input
					input_n.error += c.weight * output_n.error * sigmoid_dir(output_n.output)


			learning_rate = 1
			#Update Weights
			for n in output_nodes:
				for c in n.input_connections:
					input_ = c.input.output
					c.weight = c.weight - learning_rate * n.error * input_
					c.constrain_weight(max_weight = 4)


			for n in layer_1_nodes:
				for c in n.input_connections:
					input_ = c.input.output
					c.weight = c.weight - learning_rate * n.error * input_
					c.constrain_weight(max_weight = 4)

			for n in layer_2_nodes:
				for c in n.input_connections:
					input_ = c.input.output
					c.weight = c.weight - learning_rate * n.error * input_
					c.constrain_weight(max_weight = 4)

			for n in layer_3_nodes:
				for c in n.input_connections:
					input_ = c.input.output
					c.weight = c.weight - learning_rate * n.error * input_
					c.constrain_weight(max_weight = 4)

			for n in input_nodes:
				for c in n.input_connections:
					input_ = c.input.output
					c.weight = c.weight - learning_rate * n.error * input_
					c.constrain_weight(max_weight = 4)




			#reset error
			for training_n in training_nodes: #output neuron
				for c in training_n.input_connections: #first, reset the error before we sum the error
					input_n = c.input #input neuron
					input_n.error = 0
			# Hidden Layer Error
			for output_n in output_nodes:
				for c in output_n.input_connections:
					input_n = c.input
					input_n.error = 0
			for hidden_n in layer_3_nodes:
				for c in hidden_n.input_connections:
					input_n = c.input
					input_n.error = 0

			for hidden_n in layer_2_nodes:
				for c in hidden_n.input_connections:
					input_n = c.input
					input_n.error = 0
			# Input layer error
			for hidden_n in layer_1_nodes:
				for c in hidden_n.input_connections:
					input_n = c.input
					input_n.error = 0


def draw():
	global t
	global sio
	t += 0.1
	background(255)
	stroke(0)

	#Update Inputs
	input_1.output = random.choice(data_df['flux_normalized'].values)
	input_2.output = random.choice(data_df['flux_normalized'].values)
	input_3.output = random.choice(data_df['flux_normalized'].values)
	
	#Forward Pass
	for n in layer_1_nodes:
		n.weight_sum = 0
		for c in n.input_connections:
			n.sum_inputs(input_ = c.input.output, weight = c.weight)
		n.activate()

	for n in layer_2_nodes:
		n.weight_sum = 0
		for c in n.input_connections:
			n.sum_inputs(input_ = c.input.output, weight = c.weight)
		n.activate()

	for n in layer_3_nodes:
		n.weight_sum = 0
		for c in n.input_connections:
			n.sum_inputs(input_ = c.input.output, weight = c.weight)
		n.activate()

	for n in output_nodes:
		n.weight_sum = 0
		for c in n.input_connections:
			n.sum_inputs(input_ = c.input.output, weight = c.weight)
		n.activate()


	# Draw Nodes & Connections
	for n in input_nodes:
		n.draw(r = 0, g = 255, b = 0)
		for c in n.input_connections:
			c.draw()

	for n in layer_1_nodes:
		n.draw(r = 0, g = 0, b = 0)
		d = { 'node':n.output}
		sio.emit('dictionary', d)
		for c in n.input_connections:
			c.draw()
	for n in layer_2_nodes:
		n.draw(r = 0, g = 0, b = 0)
		d = { 'node':n.output}
		sio.emit('dictionary', d)
		for c in n.input_connections:
			c.draw()
	for n in layer_3_nodes:
		n.draw(r = 0, g = 0, b = 0)
		d = { 'node':n.output}
		sio.emit('dictionary', d)	
		for c in n.input_connections:
			c.draw()

	for n in output_nodes:
		n.draw(r = 255, g = 0, b = 0)
		d = {'error':n.error, 'output':n.output}
		sio.emit('dictionary', d)
		for c in n.input_connections:
			c.draw()
	for n in training_nodes:
		n.draw(r = 0, g = 128, b = 128)
		for c in n.input_connections:
			c.draw()
run()