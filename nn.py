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
		self.output=0  #
		self.input=0   #X value
		self.verbose=True
		self.filters = []
		for i in range(num_filters) :
			filter= 2*np.random.random((filter_size,filter_size)) - 1
			self.filters.append(filter)
		

	def __repr__(self) :
		return self.name


    
	def maxpool(self, cz, szx, szy) :
		data = self.input

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

	# That very smart code was taken from http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python/30110497
	@staticmethod
	def im2col(array, size, stride=1, padding=0):
		# Add padding to our array
		#padded_array = np.pad(array,((0, 0), (0, 0), (padding, padding), (padding, padding)),	mode='constant')
		# Get the shape of our newly made array
		#H,W = np.shape(padded_array)
		H,W = np.shape(array)
		# Get the extent
		extent = H - size + 1

		# Start index
		start_idx = np.arange(size)[:, None] * H + np.arange(size)
		offset_idx = np.arange(extent)[:, None] * H + np.arange(extent)

		return np.take(
		array, 
		np.ravel(start_idx)[:, None] + np.ravel(offset_idx)[::stride]
		)

	def cifar_rgb_to_grayscale(image):
		red = np.reshape(image[0:1024], (32, 32))
		green = np.reshape(image[1024:2048], (32, 32))
		blue = np.reshape(image[2048:3072], (32, 32))

		return 0.2989 * red + 0.5870 * green + 0.1140 * blue

		
	def max_pool2(self, inputs, size, stride=1, padding=0):
		"""
		    Description: Max pool layer
		    Parameters:
			inputs -> The input of size [batch_size] x [filter] x [shape_x] x [shape_y]
			size -> The size of the tiling
			stride -> The applied translation at each step
			padding -> The padding (padding with 0 so the last column isn't left out)
		"""

		inp_sp = np.shape(inputs)
		# We reshape it so every filter is considered an image.
		tile_col = self.im2col(inputs, size, stride=stride, padding=padding)
		# We take the max of each column
		max_ids = np.argmax(tile_col, axis=0)
		# We get the resulting 1 x 10240 vector
		result = tile_col[max_ids, range(max_ids.size)]



		new_size = (inp_sp[0] - size + 2 * padding) / stride + 1
		
		if (self.verbose) : print "res=",result, result.shape, new_size

		result = np.reshape(result, (new_size, new_size))



		# Make it from 16 x 16 x 10 to 10 x 16 x 16
		#return np.transpose(result, (2, 0, 1))
		return result

	def avg_pool2(self, inputs, size, stride, padding):
		"""
		    (Copy & paste of the max pool code with np.mean instead of np.argmax)
		    Description: Average pool layer
		    Parameters:
			inputs -> The input of size [batch_size] x [filter] x [shape_x] x [shape_y]
			size -> The size of the tiling
			stride -> The applied translation at each step
			padding -> The padding (padding with 0 so the last column isn't left out)
		"""

		inp_sp = np.shape(inputs)
		tile_col = self.im2col(reshaped, size, stride=stride, padding=padding)
		max_ids = np.mean(tile_col, axis=0)
		result = tile_col[max_ids, range(max_ids.size)]
		new_size = (inp_sp[2] - size + 2 * padding) / stride + 1
		result = np.reshape(result, (new_size, new_size, inp_sp[0]))
		return np.transpose(result, (2, 0, 1))

	def convolve(self, inputs, filter, stride=1, padding=0):
		"""
		    Description: Convolution layer
		"""

		kernel_size = filter.shape[0]
		new_size = (np.shape(inputs)[1] - kernel_size + 2 * padding) / stride + 1
		tile_col = self.im2col(inputs, kernel_size, stride, padding)
		kernel_col = np.reshape(filter, -1)
		result = np.dot(kernel_col, tile_col)
		return np.reshape(result, (new_size, new_size))

	def conv_backward(self, dH):
		'''
		The backward computation for a convolution function

		Arguments:
		dH -- gradient of the cost with respect to output of the conv layer (H), numpy array of shape (n_H, n_W) assuming channels = 1

		Returns:
		dX -- gradient of the cost with respect to input of the conv layer (X), numpy array of shape (n_H_prev, n_W_prev) assuming channels = 1
		dW -- gradient of the cost with respect to the weights of the conv layer (W), numpy array of shape (f,f) assuming single filter
		'''

		# Retrieving information from the "cache"
		(X, W) = self.cache

		# Retrieving dimensions from X's shape
		(n_H_prev, n_W_prev) = X.shape

		# Retrieving dimensions from W's shape
		(f, f) = W.shape

		# Retrieving dimensions from dH's shape
		(n_H, n_W) = dH.shape

		# Initializing dX, dW with the correct shapes
		dX = np.zeros(X.shape)
		dW = np.zeros(W.shape)

		# Looping over vertical(h) and horizontal(w) axis of the output
		for h in range(n_H):
			for w in range(n_W):
				dX[h:h+f, w:w+f] += W * dZ(h,w)
				dW += X[h:h+f, w:w+f] * dZ(h,w)

		return dX, dW

	def forward(self, x) :
		"""forward pass using x as input"""
		self.input = x
		output=list()
		for f in self.filters:
			r = self.convolve(x, f)
			print(r)
			output.append(r)
		return output
 
	def gradient(self, error) :
        	"""calculate gradient using error"""
		error = self.conv_backward(error)
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


