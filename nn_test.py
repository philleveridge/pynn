from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

#from nn import Layer
#from nn import Network
#from nn import Activation
#from nn import CNN
#from nn import RNN

from nn import *

import numpy as np


#test1=True
#test2=True
#test3=True
#test4=True
#test5=True
#test6=True
test7=True

if 'test1' in vars() :
	print "Test 1 ========================================================================"
	np.random.seed(0)
	n=Network([	Layer(3,4, "1"), 
			Layer(4,1, "2")])

	n.train(60000, 
		np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1] ]), 
		np.array([[0,1,1,0]]).T)

	print("Evaluating:")
	print("  0 0 1 = " + n.eval(np.array([0,0,1]), 0))
	print("  0 1 1 = " + n.eval(np.array([0,1,1]), 1))
	print("  1 0 1 = " + n.eval(np.array([1,0,1]), 1))
	print("  1 1 1 = " + n.eval(np.array([1,1,1]), 0))

if 'test2' in vars():
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


if 'test3' in vars():
	print "Test 3 ========================================================================"

	# Helper function to plot a decision boundary.
	# If you don't fully understand this function don't worry, it just generates the contour plot below.
	def plot_decision_boundary(pred_func):
	    # Set min and max values and give it some padding
	    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	    h = 0.01
	    # Generate a grid of points with distance h between them
	    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	    # Predict the function value for the whole gid
	    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	    Z = Z.reshape(xx.shape)
	    # Plot the contour and training examples
	    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, axis=0)
	    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

	def get_test_data () :       
	    # Generate a dataset and plot it
	    np.random.seed(0)
	    X, y = datasets.make_moons(200, noise=0.20)
	    y = np.array([ [x,1-x] for x in y])
	    return X,y

	np.random.seed(0)
	n=Network([	Layer(2,3,"Input",  afn=Activation.tanh), 
			Layer(3,2,"Hidden", afn=Activation.softmax)], epsilon=0.01)

	X,y = get_test_data ()
	n.train(20000, X, y)

	# Plot the decision boundary
	y=y[:,1]
	plot_decision_boundary(lambda x: n.predict(x))
	plt.title("Decision Boundary for hidden layer size 3")
	plt.show()



if 'test4' in vars():
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

if 'test5' in vars():
	print "Test 5 === CNN test on image ================================================="
	np.random.seed(0)
	
	def unpickle(file):
	    import pickle
	    with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	    return dict        
                
	#d = unpickle('cifar-10-python.tar.gz')

                
if 'test6' in vars():
	print "Test 6 === RNN test ====================================================="
	np.random.seed(0)       
    
	n=Network([	RNN  (3,4, "rnn"), 
			Layer(4,1, "output")])


if 'test7' in vars():
	print "Test 7 =========================================================="
	np.random.seed(0)  

	dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
	# split into input (X) and output (Y) variables
	X = dataset[:,0:8]
	Y = dataset[:,8]   

	n=Network([	Layer(8, 12, "input"), 
			Layer(12,8,  "hidden"),
			Layer(8, 1,  "2")])

	n.train_batch(6000, X, Y, batch_size=150)

  
