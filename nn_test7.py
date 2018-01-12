from nn import *

import numpy as np


print "Test 7 =========================================================="
np.random.seed(0)  

dataset = np.loadtxt("data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]   

def initweights(a,b) :
    return 0.05*np.random.random((a,b))

n=Network([	Layer(8, 12, "input",  afn=Activation.relu, iwfn=initweights), 
		Layer(12,8,  "hidden", afn=Activation.relu, iwfn=initweights),
		Layer(8, 1,  "2")])
n.verbose=True
n.epoc_print = 15
n.cost_fn = CostFunctions.binary_cross_entropy_cost
n.train_batch(150, X, Y, batch_size=30)

  
