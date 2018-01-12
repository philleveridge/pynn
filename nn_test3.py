from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt


from nn import *

import numpy as np



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


