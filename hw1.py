import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt

def linear_gd(X,Y,lrate=0.1,num_iter=1000):
    # return parameters as numpy array
    bias_column = np.ones((X.shape[0], 1))
    X = np.hstack((bias_column, X))
    n, d = X.shape
    w = np.zeros(d)
    for i in range(num_iter):
        gradient = np.dot(X.T, (np.dot(X, w) - Y)) / n
        w -= lrate * gradient
    return w

def linear_normal(X,Y):
    # return parameters as numpy array
    bias_column = np.ones((X.shape[0], 1))
    X = np.hstack((bias_column, X))
    XT_X = np.dot(X.T, X)
    XT_X_inv = np.linalg.inv(XT_X)
    XT_X_inv_XT = np.dot(XT_X_inv, X.T)
    w = np.dot(XT_X_inv_XT, Y)
    return w

def plot_linear():
    X, Y = utils.load_reg_data()
    print(X)
    w = linear_normal(X, Y)
    line_X = np.linspace(min(X), max(X), 100)
    print(w)
    line_Y = w[1] * line_X + w[0]
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(line_X, line_Y, color='red', label='Linear Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return 

if __name__ == "__main__":
    plot_linear()