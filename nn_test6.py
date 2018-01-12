from nn import *

import numpy as np


print "Test 6 === RNN test ====================================================="
np.random.seed(0)       

n=Network([	RNN  (3,4, "rnn"), 
		Layer(4,1, "output")])


