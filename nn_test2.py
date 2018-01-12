

from nn import *

import numpy as np




print "Test 2 ========================================================================"

np.random.seed(0)
n=Network([	Layer(2,20, "1", afn=Activation.tanh), 
		Layer(20,1 , "2", afn=Activation.tanh)])
n.verbose=True
n.train(60000, 
	np.array([[0,0],[0,1],[1,0],[1,1] ]), 
	np.array([[0,1,1,0]]).T)

print("Evaluating:")
print("  0 0 = " + n.eval(np.array([0,0]), 0))
print("  0 1 = " + n.eval(np.array([0,1]), 1))
print("  1 0 = " + n.eval(np.array([1,0]), 1))
print("  1 1 = " + n.eval(np.array([1,1]), 0))


