from nn import *
import numpy as np
import pickle

print "Test 5 === CNN test on image ================================================="
np.random.seed(0)

def unpickle(file):

    with open(file, 'rb') as fo:
	dict = pickle.load(fo, encoding='bytes')
    return dict        
        
#d = unpickle('cifar-10-python.tar.gz')

