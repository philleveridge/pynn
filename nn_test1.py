from nn import *

import numpy as np


print "Test 1 ========================================================================"
np.random.seed(0)
n=Network([	Layer(3,4, "1"), 
		Layer(4,1, "2")])
n.verbose = True
n.epoc_print = 10000

"""
n.train(60000, 
	np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1] ]), 
	np.array([[0,1,1,0]]).T)

print("Evaluating:")
print("  0 0 1 = " + n.eval(np.array([0,0,1]), 0))
print("  0 1 1 = " + n.eval(np.array([0,1,1]), 1))
print("  1 0 1 = " + n.eval(np.array([1,0,1]), 1))
print("  1 1 1 = " + n.eval(np.array([1,1,1]), 0))
"""

print "Test 1 ========================================================================"
np.random.seed(0)
n=Network([	Layer(3,4, "1"), 
		Layer(4,1, "2")])
n.verbose = True
n.epoc_print = 10000
#n.cost_fn=CostFunctions.binary_cross_entropy_cost
n.train_batch(60000, 
	np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1] ]), 
	np.array([[0,1,1,0]]).T, batch_size=2)

print("Evaluating:")
print("  0 0 1 = " + n.eval(np.array([0,0,1]), 0))
print("  0 1 1 = " + n.eval(np.array([0,1,1]), 1))
print("  1 0 1 = " + n.eval(np.array([1,0,1]), 1))
print("  1 1 1 = " + n.eval(np.array([1,1,1]), 0))

