import numpy as np
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Note: not all helper functions here are needed for the homework,
# but feel free to play around with them.

def load_reg_data():
    # load the regression synthetic data
    np.random.seed(1) # force seed so same data is generated every time
    X = np.linspace(0,4,num=100)
    noise = np.random.normal(size=X.shape, scale=0.4)
    w = 0.5
    b = 1.
    Y = w * X**2 + b + noise
    X = np.reshape(X,[-1,1])
    return X,Y

def load_xor_data():
    X = np.array([[-1,1],[1,-1],[-1,-1],[1,1]])
    Y = np.prod(X,axis=1)
    return X,Y

def load_nn_data():
    np.random.seed(3)
    X1,Y1 = linear_problem(np.array([-1,0.3]), margin=1.0, size=100)
    X2,Y2 = linear_problem(np.array([1,-0.1]), margin=6.0, size=100, bounds=[-7,7],trans=3)
    X = np.concatenate([X1,X2], axis=0)
    Y = np.concatenate([Y1,Y2],axis=0)
    return X,Y

def load_iris_data():
    X,Y = load_iris(return_X_y=True)
    np.random.seed(1)
    s = np.arange(len(X))
    np.random.shuffle(s)
    return X[s],Y[s]

def load_logistic_data():
    np.random.seed(1) # reset seed
    return linear_problem(np.array([-1.,2.]),margin=1.5,size=200)

def contour_plot(xmin, xmax, ymin, ymax, M, ngrid = 33):
    """
    make a contour plot without
    @param xmin: lowest value of x in the plot
    @param xmax: highest value of x in the plot
    @param ymin: ditto for y
    @param ymax: ditto for y
    @param M: prediction function, takes a (X,Y,2) numpy ndarray as input and returns an (X,Y) numpy ndarray as output
    @param ngrid:
    """
    xgrid = np.linspace(xmin, xmax, ngrid)
    ygrid = np.linspace(ymin, ymax, ngrid)
    (xx, yy) = np.meshgrid(xgrid, ygrid)
    D = np.dstack((xx, yy))
    zz = M(D)
    C = plt.contour(xx, yy, zz,
                    cmap = 'rainbow')
    plt.clabel(C)
    plt.show()

def linear_problem(w,margin,size,bounds=[-5.,5.],trans=0.0):
    in_margin = lambda x: np.abs(w.dot(x)) / np.linalg.norm(w) < margin
    X = []
    Y = []
    for i in range(size):
        x = np.random.uniform(bounds[0],bounds[1],size=[2]) + trans
        while in_margin(x):
            x = np.random.uniform(bounds[0],bounds[1],size=[2]) + trans
        if w.dot(x) + trans > 0:
            Y.append(1.)
        else:
            Y.append(-1.)
        X.append(x)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y
