import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:13:42 2017

@author: phil
"""

import numpy as np

class Activation:
	# Activation functions
	@staticmethod
	def sigmoid(x, deriv=False) :
		if (deriv==True) :
			return (x* (1 - x))
		return 1/(1+np.exp(-x))

	@staticmethod
	def tanh(x, deriv=False):
	  if deriv==True:
	      return 1 - x ** 2     
	  return np.tanh(x)
		
	@staticmethod
	def softmax(x, deriv=False):
	    """Compute softmax values for each sets of scores in x."""
	    if deriv==True:
		return x*(1-x)
	    t=np.exp(x)
	    return t / np.sum(t, axis=1, keepdims=True)

	@staticmethod
	def relu(x, deriv=False):
	  if deriv==True:
	      return 1*(x>0)     
	  return np.maximum(x,0,x)

class CostFunctions:
	@staticmethod
	def sum_squared_error( outputs, targets, deriv=False ):
	    if deriv == True:
		return outputs - targets 
	    else:
		return 0.5 * np.mean(np.sum( np.power(outputs - targets,2), axis = 1 ))


	@staticmethod
	def binary_cross_entropy_cost( outputs, targets, deriv=False, epsilon=0.0001):
	    """
	    The output should be in the range [0, 1]
	    """
	    # Prevent overflow
	    outputs = np.clip(outputs, epsilon, 1 - epsilon)
	    divisor = np.maximum(outputs * (1 - outputs), epsilon)
	    
	    if deriv == True :
		return (outputs - targets) / divisor
	    else:
		return np.mean(-np.sum(targets * np.log( outputs ) + (1 - targets) * np.log(1 - outputs), axis=1))


class Layer:
	#initialise
	def __init__(self, num_inputs, num_outputs, layer_name, afn=Activation.sigmoid, iwfn=None) :
		if iwfn==None :
			self.syn = 2*np.random.random((num_inputs,num_outputs)) - 1
		else:
			self.syn = iwfn(num_inputs,num_outputs)                
    		self.bias = np.zeros((1,num_outputs))
		self.name = layer_name	
		self.afn = afn
		self.output=0

	def __repr__(self) :
		print  "L{} {}".format(self.name, self.output)
		print  "W{} {}".format(self.name, self.syn)
  		if self.delta in vars() :
                 print  "D{} {}".format(self.name, self.delta)

	def forward(self, input) :
         op=self.afn(np.dot(input,self.syn)+self.bias)
         self.output = op
         return self.output

	def gradient(self, error) :
		self.delta = error * self.afn(self.output, deriv=True)
		error  = self.delta.dot(self.syn.T)
		return error

	def update_weight(self, input, learning_rate) :
		self.syn += input.T.dot(self.delta) * learning_rate
		self.bias += learning_rate*np.sum(self.delta, axis=0)
		return self.output




class RNN(Layer):
	def __init__(self, num_inputs, num_outputs, layer_name, afn=None) :
		self.name = layer_name
		self.afn  = afn
		self.no   = num_outputs
		self.ni   = num_inputs
		self.W_xh = 2*np.random.random((num_inputs,num_outputs)) - 1
		self.W_hh = 2*np.random.random((num_inputs,num_outputs)) - 1
		self.output=0

	def forward(self, x):
		# update the hidden state		
		if self.h in vars() :
			self.output = afn(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
		else :
			self.output = afn(np.dot(self.W_xh, x))		
		return self.output

	def gradient(self, error) :
		pass

	def update_weight(self, input, learning_rate) :
		pass


class CNN(Layer):
	def __init__(self, num_filters, filter_size, layer_name, afn=None) :
		self.name = layer_name
		self.afn  = afn
		self.convolve = True
		self.output=0
		self.filters = []
		for i in range(num_filters) :
			self.filters.append(2*np.random.random((filter_size,filter_size)) - 1)
		

	def __repr__(self) :
		return self.name
    
	def maxpool(self, data, cz, szx, szy) :
		w=data.shape[1]
		h=data.shape[0]
		dout = np.zeros([h//szx,w//szy])   
		print("maxpool",w,h,cz,szx,szy)         
		for i in range(0,w//szy) :
			for j in range(0,h//szx) :
			   try:
			       dout[j,i] = data[j*szx:j*szx+cz,i*szy:i*szy+cz].max()
			   except:
			       print("err",i,j)
			       pass
		self.output = dout
		return dout

	def conv(self, data, filter, szx, szy) :
		w=data.shape[1]
		h=data.shape[0]
		cz=filter.shape[0]
		dout = np.zeros(data.shape)   
		print("conv",w,h,cz,szx,szy)        
		try:
			for i in range(0,w-cz+1) :
			    for j in range(0,h-cz+1) :
				#print(i,j,i*szy,j*szx)
				dout[j*szx:j*szx+cz,i*szy:i*szy+cz] +=data[j*szx:j*szx+cz,i*szy:i*szy+cz]*filter
				#print(i,j,dout)
		except:
			print("err",i,j)
		self.output = dout
		return dout

	def forward(self, x) :
        	"""forward pass using x as input"""
		for f in filters:
			pass
		self.output = x
 
	def gradient(self, error) :
        	"""calculate gradient using error"""
		self.delta =0
		return error

	def update_weight(self, input, learning_rate) :
        	"""update weights/filters using gradient"""
		for f in filters:
			pass  	

class Network:
	def __init__(self, layers, epsilon=0.1, lmda=0.01) :
		self.layers = layers
		self.epsilon = epsilon
		self.lmda   = lmda
		self.verbose=False
		self.epoc_print=200
		self.cost_fn = CostFunctions.sum_squared_error
		
	def __repr__(self):
		for l in self.layers :
			print(l)
	
	def forward(self, data):
		output = data
		for l in self.layers :
			output = l.forward(output)
		return output	

	def eval(self, x, y):
		output = self.forward(x)
		normalized = int(round(output[0]))
		error = y - output[0]
		return "%d (% .3f)   Error: %.3f" % (normalized, output[0], error)

	def predict(self, x):
		probs = self.forward(x)
		return np.argmax(probs, axis=1)

	def backprop(self, error, data):
		for l in reversed(self.layers) :
			error = l.gradient(error)

		for l in self.layers :
			data = l.update_weight(data, self.epsilon)		

	def train_batch(self, no_epocs, input_data, input_labels, batch_size=0) :
		if batch_size==0 or batch_size>len(input_data) :
			batch_size=len(input_data)	
   
		noi =input_data.shape[1]

		if len(input_labels.shape)==1 :
			nol =1
		else:
			nol = input_labels.shape[1]

		for j in range(0,no_epocs) :
			for bn in range(0,len(input_data)//batch_size) :
				idx = bn*batch_size

				data   = input_data  [idx:idx+batch_size] 
				labels = input_labels[idx:idx+batch_size] 
				#data.shape = (batch_size,noi)
				labels.shape = (batch_size,nol)
	    
				pred = self.forward(data)
				error = self.cost_fn(labels, pred, deriv=True)
				self.backprop(error, data)				

			if self.verbose and j%self.epoc_print == 0 :
				cost = self.cost_fn(labels, pred)
				print "epoc: {}  : loss {}".format(j, cost)


	def train(self, n, input_data, input_labels) :
		if len(input_labels.shape)==1 :
			input_labels.shape = (input_labels.shape[0],1)

		for j in range(0,n) :
			pred = self.forward(input_data)
			error = self.cost_fn(input_labels, pred, deriv=True)
			self.backprop(error, input_data)

			if self.verbose and j%self.epoc_print==0 :
				cost = self.cost_fn(input_labels, pred)
				print "epoc: {}  loss {}".format(j, cost)


			
if __name__ == "__main__" :
    print("This is a library")


