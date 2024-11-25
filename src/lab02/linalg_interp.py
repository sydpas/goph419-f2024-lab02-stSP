import numpy as np


def gauss_iter_solve(A, b, x0, tol, alg):
    """
    Definition:
    -----
    This function solves a linear system of equations A * x = b using the Gauss-Seidel approach.
    
    Parameters:
    -----
    A: array_like
        The coefficient matrix.
    b: array_like
        The right-hand-side vector(s).
    x0: (optional) array_like, shape of b OR single column with same rows as A and b
        The initial guess(es). Has a default value of None.
    tol: (optional) float
        Gives the relative error tolerance (the stopping criterion). Has a default value of 1e-8.
    alg: (optional) str flag
        Has a default value of 'seidel' or 'jacobi' based on the algorithm used.
        
    Returns:
    -----
    x: numpy.ndarray, shape of b.
    """

    n = len(b)  # number of rows
    max_iter = 1000  # maximum iterations to determine convergence

    x = np.asarray(x0, dtype=float)  # converts x0 into an array if not already and stores in x
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if x.shape != b.shape:
        raise ValueError("x0 must have the same shape as b.")
    if x.shape[0] != A.shape[0] and b.shape[0]:
        raise ValueError("x0 must have the rows as A and b.")

    for iteration in range(max_iter):
        x_new = x.copy()  # x_new holds a copy of x
        for k in range(n):
            a_row = A[k, :]  # each k, a_row will be a new row from A.
            kp1 = (k + 1)
            # isolate k. b[k] is from b at row k. subtract previous values, subtract remaining values, divide by diag.
            x[k] = (b[k] - a_row[:k] @ x_new[:k] - a_row[kp1:] @ x[kp1:]) / A[k, k]
        if np.linalg.norm(x_new - x) < tol:  # check for convergence of x_new compared to x
            return x


def spline_function(xd, yd, order):
    """
    Definition:
    -----
    This function generates a spline function given two vectors x and y.

    Parameters:
    -----
    xd: array_like, float
        Increases in value.
    yd: array_like, float, shape of xd
    order: (optional) int
        Possible values are 1, 2, and 3 with a default value of 3.

    Returns:
    -----
    function: Takes one parameter and returns the interpolated y-value(s).
    """