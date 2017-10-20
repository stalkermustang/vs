#!/usr/bin/env python
# -*- coding: utf8 -*-


####################################################
### You are not allowed to import anything else. ###
####################################################

import numpy as np


def power_sum(l, r, p=1.0):
    """
        input: l, r - integers, p - float
        returns sum of p powers of integers from [l, r]
    """
    return np.sum([i**p for i in range(l, r+1, 1)])

def replace_outliers(x, std_mul=3.0):
    """
        input: x - numpy vector, std_mul - positive float
        returns copy of x with all outliers (elements, which are beyond std_mul * (standart deviation) from mean)
        replaced with mean  
    """
    x = np.array(x, dtype=float)
    my_std = np.std(x)
    my_mean = np.mean(x)
    x = np.array([my_mean if (i>my_mean+std_mul*my_std or i<my_mean-std_mul*my_std) else i for i in x ])
    return x

def get_eigenvector(A, alpha):
    """
        input: A - square numpy matrix, alpha - float
        returns numpy vector - any eigenvector of A corresponding to eigenvalue alpha, 
                or None if alpha is not an eigenvalue.
    """
    my_vals = np.linalg.eigvals(A)
    if alpha in my_vals:
        return np.linalg.eig(A)[1][0]
    else:
        return None

def discrete_sampler(p):
    """
        input: p - numpy vector of probability (non-negative, sums to 1)
        returns integer from 0 to len(p) - 1, each integer i is returned with probability p[i] 
    """
    return int(np.random.choice([x for x in range(len(p))], p=p, size=1))

def gaussian_log_likelihood(x, mu=0.0, sigma=1.0):
    """
        input: x - numpy vector, mu - float, sigma - positive float
        returns log p(x| mu, sigma) - log-likelihood of x dataset 
        in univariate gaussian model with mean mu and standart deviation sigma
    """
    return np.sum([ -0.5*np.log(2*np.pi*sigma**2)-0.5*(((i-mu)**2)/sigma**2) for i in x])

def gradient_approx(f, x0, eps=1e-8):
    """
        input: f - callable, function of vector x. x0 - numpy vector, eps - float, represents step for x_i
        returns numpy vector - gradient of f in x0 calculated with finite difference method 
        (for reference use https://en.wikipedia.org/wiki/Numerical_differentiation, search for "first-order divided difference")
    """
    x0 = np.array(x0, dtype=float)
    res = np.array([],dtype=float)
    for i in range(len(x0)):
        x0h=np.array(x0)
        x0h[i] = x0h[i] + eps
        res = np.append(res, (f(x0h)-f(x0))/eps)

    return res

def gradient_method(f, x0, n_steps=1000, learning_rate=1e-2, eps=1e-8):
    """
        input: f - function of x. x0 - numpy vector, n_steps - integer, learning rate, eps - float.
        returns tuple (f^*, x^*), where x^* is local minimum point, found after n_steps of gradient descent, 
                                        f^* - resulting function value.
        Impletent gradient descent method, given in the lecture. 
        For gradient use finite difference approximation with eps step.
    """
    x0 = np.array(x0, dtype=float)
    for _ in range(n_steps):
        my_grad = gradient_approx(f, x0, eps=eps)
        x0-= learning_rate*my_grad
    return (f(x0), x0)

def linear_regression_predict(w, b, X):
    """
        input: w - numpy vector of M weights, b - bias, X - numpy matrix N x M (object-feature matrix), 
        N - number of objects, M - number of features.
        returns numpy vector of predictions of linear regression model for X
        https://xkcd.com/1725/
    """
    preds = np.array([], dtype=float)
    for i in X:
        pd = b + np.dot(i, w)
        preds = np.append(preds, pd)
    return preds

def mean_squared_error(y_true, y_pred):
    """
        input: two numpy vectors of object targets and model predictions.
        return mse
    """
    err = 0.
    for i, j in zip(y_true, y_pred):
        err += (i-j)**2
    return err/len(y_true)

def linear_regression_mse_gradient(w, b, X, y_true):
    """
        input: w, b - weights and bias of a linear regression model,
                X - object-feature matrix, y_true - targets.
        returns gradient of linear regression model mean squared error w.r.t (with respect to) w and b
    """
    val = np.append(w, b)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    return 2./len(X)*(X.T.dot(X.dot(val)-y_true))

class LinearRegressor:
    def fit(self, X_train, y_train, n_steps=1000, learning_rate=1e-2, eps=1e-8):
        """
           input: object-feature matrix and targets.
           optimises mse w.r.t model parameters
       """
        self.w = np.array([0.] * X_train.shape[1])
        self.b = 0.
        self.p_mse = np.array([mean_squared_error(y_train, linear_regression_predict(self.w, self.b, X_train))])
        for _ in range(n_steps):
            gd_r = linear_regression_mse_gradient(self.w, self.b, X_train, y_train)
            gd_w = np.array(gd_r[:-1])
            gd_b = gd_r[-1]
            w_upd = np.array(learning_rate*gd_w)
            self.b -= learning_rate * gd_b
            self.w -= np.ravel(w_upd)
            c_mse = mean_squared_error(y_train, linear_regression_predict(self.w, self.b, X_train))
            if np.abs(self.p_mse[-1]-c_mse) < eps:
                self.p_mse = np.append(self.p_mse, c_mse)
                break
            else:
                self.p_mse = np.append(self.p_mse, c_mse)
       
        return self
 
 
    def predict(self, X):
        return linear_regression_predict(self.w, self.b, X)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_der(x):
    """
        returns sigmoid derivative w.r.t. x
    """
    return sigmoid(x)*(1.-sigmoid(x))

def relu(x):
    return np.maximum(x, 0)

def relu_der(x):
    """
        return relu (sub-)derivative w.r.t x
    """
    return (x >= 0) * 1