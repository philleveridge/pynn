from nn import *

import numpy as np


print "Test 4 =====convolve and maxpool test ========================================="

np.random.seed(0)

data = np.array([ [1,2,2,1,1,1,2,0],
	          [1,7,3,1,1,0,2,5],
	          [5,2,5,1,1,0,2,3],
	          [1,3,9,2,1,4,2,5],
	          [4,0,3,7,1,0,1,3],
	          [1,6,3,1,8,4,2,1],
	          [2,0,8,1,1,0,2,5],
	          [1,9,6,1,1,7,2,4]])

filter = np.array([[ 0, 0, 0],
	          [ 0, 1, 0],
            [ 0, 0, 0]])

filter1 = np.array([[ 0, 1 ,0],
	           [ 0, 1, 0],
	           [ 0, 1, 0]])

n=Network([	CNN(2,3, "image"),   #2 feature 3x3
	Layer(4,1, "output")])

#overwrite random init of filters
n.verbose=True
n.layers[0].filters[0] = filter             
n.layers[0].filters[1] = filter1
     
print ("out=",n.layers[0].maxpool(data,4,2,2))
print ("out=",n.layers[0].conv(data,filter,1,1) )

#o = n.train(5000,data, y? )

