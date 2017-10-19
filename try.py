import numpy as np

def my_f(x):
    return 3*x[0]**2+17*x[1]**2

def gradient_approx(f, x0, eps=1e-8):
    """
        input: f - callable, function of vector x. x0 - numpy vector, eps - float, represents step for x_i
        returns numpy vector - gradient of f in x0 calculated with finite difference method 
        (for reference use https://en.wikipedia.org/wiki/Numerical_differentiation, search for "first-order divided difference")
    """
    #x0 = np.array(x0, dtype=float)
    res = np.array([],dtype=float)
    for i in range(len(x0)):
        x0h=np.array(x0)
        x0h[i]+=eps
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
    for i in range(n_steps):
        my_grad = gradient_approx(f, x0, eps=eps)
        x0-= learning_rate*my_grad
    return (f(x0), x0)

def linear_regression_mse_gradient1(w, b, X, y_true):
    """
        input: w, b - weights and bias of a linear regression model,
                X - object-feature matrix, y_true - targets.
        returns gradient of linear regression model mean squared error w.r.t (with respect to) w and b
    """
    val = np.append(w, b)
    X = np.hstack((X, np.ones((X.shape[0],1))))
    return 2./len(X)*(X.T.dot(X.dot(val)-y_true))

print(gradient_method(my_f, np.array([10,15])))