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

    A = np.asarray(A, dtype=float) # converting A into an array if not already.
    b = np.asarray(b, dtype=float) # converting b into an array if not already.
    if x0 is None:
        x0 = np.zeros_like(b, dtype=float) # initializing x0 if it is not given values.
    else:
        x0 = np.asarray(x0, dtype=float) # converting x0 into an array if not already.

    n = len(b)  # number of rows
    max_iter = 2000  # maximum iterations to determine convergence

    x = np.asarray(x0, dtype=float)  # converts x0 into an array if not already and stores in x
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if x.shape != b.shape:
        raise ValueError("x0 must have the same shape as b.")
    if x.shape[0] != A.shape[0] and b.shape[0]:
        raise ValueError("x0 must have the rows as A and b.")

    if alg == 'seidel':
        print(f'Algorithm used: Gauss-Seidel.')
        for iteration in range(max_iter):
            x_new = x.copy()  # x_new holds a copy of x
            for k in range(n):
                a_row = A[k, :]  # each k, a_row will be a new row from A.
                kp1 = (k + 1)
                # isolate k. b[k] is from b at row k. sub previous values, subtract remaining values, divide by diag.
                x[k] = (b[k] - a_row[:k] @ x_new[:k] - a_row[kp1:] @ x[kp1:]) / A[k, k]
            if np.linalg.norm(x_new - x) < tol:  # check for convergence of x_new compared to x
                return x
    elif alg == 'jacobi':
        print(f'Algorithm used: Jacobi.')
        for iteration in range(max_iter):
            x_new = x.copy()
            for k in range(n):
                a_row = A[k, :]
                kp1 = (k + 1)
                # find x_new using the previous values
                x_new[k] = (b[k] - np.dot(a_row[:k], x[:k]) - np.dot(a_row[kp1:], x[kp1:])) / A[k, k]
            if np.linalg.norm(x_new - x) < tol:  # check for convergence of x_new compared to x
                return x_new
            x = x_new.copy()
    elif alg != 'seidel' or 'jacobi':
        raise ValueError("Please use either Gauss-Seidel or Jacobi algorithm.")


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
    n = len(yd)-1

    # linear spline
    if order == 1:
        slope_list = np.zeros(n)
        for i in range(len(yd) - 1):
            ip1 = i + 1
            slope_list[i] = (yd[ip1] - yd[i]) / (xd[ip1] - xd[i])  # equation 24
        return slope_list

    # quadratic spline
    if order == 2:



