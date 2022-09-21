from p5 import *
import numpy as np
import random
from scipy.stats import pearsonr


class Point(object):
	def __init__(self, x, y, size):
		self.x = x
		self.y = y
		self.size = size

	def draw(self):
		# stroke_weight(self.size)
		point(self.x, self.y)

class Neuron(Point):
	def __init__(self,x,y,size):
		super(Neuron, self).__init__(x,y,size) #call child as first argument
		self.input_connections = []
		self.weight_sum = 0
		self.output = 0
		self.error = 0
		self.dirivative = 0
		self.dropout_percentage = 0
		self.dropout = 0
		self.last_n_outputs = []
		self.last_n_outputs_length = 100
		self.correlation_connections = []

		pass

	def sum_inputs(self,input_,weight):
		self.weight_sum += input_ * weight

	def activate(self):
		self.output = (1 / (1 + exp(-self.weight_sum))) #sigmoid activation
		# self.output = self.weight_sum * math.cos(self.weight_sum) #Growing Cosine Unit (GCU) activation
		
		self.last_n_outputs.append(self.output)
		if len(self.last_n_outputs) > self.last_n_outputs_length:
			self.last_n_outputs = self.last_n_outputs[1::]


	def draw(self, r, g, b):
		if self.dropout == 1:
			draw_dropout = 0
		else:
			draw_dropout = 1
		stroke_weight((self.output * self.size)**2 * draw_dropout)
		stroke(r,g,255000 * abs(self.error), 255)
		point(self.x,self.y)

class Connection(object):
	def __init__(self, input_, output):
		self.input = input_
		self.output = output
		self.weight = np.random.uniform()
		self.correlation = []
		self.correlation_history = []
		self.avg_correlation = 0 
		pass

	def constrain_weight(self, max_weight):
		if self.weight >= max_weight:
			self.weight = max_weight
		if self.weight <= -max_weight:
			self.weight = -max_weight

	def draw(self):
		stroke(0,0,0,self.weight * 255)
		stroke_weight((self.weight)**2)
		line(self.input.x, self.input.y, self.output.x, self.output.y)