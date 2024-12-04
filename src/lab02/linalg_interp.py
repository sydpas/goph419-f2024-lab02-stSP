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

    A = np.asarray(A, dtype=float)  # converting A into an array if not already.
    b = np.asarray(b, dtype=float)  # converting b into an array if not already.
    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)  # initializing x0 if it is not given values.
    else:
        x0 = np.asarray(x0, dtype=float)  # converting x0 into an array if not already.

    n = len(b)  # number of rows
    max_iter = 5000  # maximum iterations to determine convergence

    x = np.asarray(x0, dtype=float)  # stores x0 in x
    if x.shape != b.shape:
        raise ValueError("x0 must have the same shape as b.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if x.shape[0] != A.shape[0] and b.shape[0]:
        raise ValueError("x0 must have the rows as A and b.")

    if alg == 'seidel':
        for iteration in range(max_iter):
            x_new = x.copy()  # x_new holds a copy of x
            for k in range(n):
                a_row = A[k, :]  # each k, a_row will be a new row from A.
                kp1 = (k + 1)
                # isolate k. b[k] is from b at row k. sub previous values, subtract remaining values, divide by diag.
                x[k] = (b[k] - a_row[:k] @ x_new[:k] - a_row[kp1:] @ x[kp1:]) / A[k, k]
            if np.linalg.norm(x_new - x) < tol:  # check for convergence of x_new compared to x
                return x
        raise RuntimeWarning('This system has not converged.')
    elif alg == 'jacobi':
        for iteration in range(max_iter):
            x_new = np.zeros_like(x)
            for k in range(n):
                a_row = A[k, :]
                kp1 = (k + 1)
                # find x_new using the previous values
                x_new[k] = (b[k] - np.dot(a_row[:k], x[:k]) - np.dot(a_row[kp1:], x[kp1:])) / A[k, k]
            x = x_new.copy()
            if np.linalg.norm(x_new - x) < tol:  # check for convergence of x_new compared to x
                return x_new
            x = x_new.copy
        raise RuntimeWarning('This system has not converged.')
    elif alg not in ['seidel', 'jacobi']:
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
    n = len(xd) - 1  # the number of intervals

    xd = np.array(xd, dtype=float)
    yd = np.array(yd, dtype=float)

    if len(xd) != len(yd):
        raise ValueError('xd and yd do not have the same length.')
    if len(np.unique(xd)) != len(xd):
        raise ValueError('There are repeated vales in xd.')
    if not np.all(np.diff(xd) > 0):
        raise ValueError('The xd values are not in increasing order.')

    # linear spline
    if order == 1:
        def linear_spline(xi):
            for i in range(len(xd) - 1):
                if xd[i] <= xi <= xd[i + 1]:
                    ip1 = i + 1
                    slope = (yd[ip1] - yd[i]) / (xd[ip1] - xd[i])
                    yi = yd[i] + slope * (xi - xd[i])
                    return yi
            raise ValueError("x is out of bounds of the xd values.")
        return linear_spline

    # quadratic spline
    elif order == 2:
        xdiff = np.diff(xd)  # how far apart the xd values are
        ydiff = np.diff(yd)  # the difference between yd values

        # now we set up the system of equations to find second derivatives c (continuity constraints)
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)

        for i in range(1, n):
            A[i, i - 1] = xdiff[i - 1]  # 2nd derivative at previous point
            A[i, i] = 2 * (xdiff[i - 1] + xdiff[i])  # 2nd derivative at current point
            A[i, i + 1] = xdiff[i]  # 2nd derivative at next point
            # ensures the slope at the points is continuous
            b[i] = 3 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])

        # clarify boundary conditions
        A[0, 0] = 1  # sets first 2nd derivative to 0
        A[-1, -1] = 1  # sets last 2nd derivative to 0

        # solve for c and compute coefficients
        c = np.linalg.solve(A, b)
        a_coef = (c[1:] - c[:-1]) / (3 * xdiff)
        b_coef = (ydiff / xdiff) - xdiff * (c[1:] + 2 * c[:-1]) / 3

        # now we compute the spline
        def quad_spline(xi):
            i = np.searchsorted(xd, xi) - 1  # searching the interval
            i = np.clip(i, 0, n - 1)  # make sure the i is within bounds
            xdiff = xi - xd[i]  # the difference between xi and the LHS
            yi = yd[i] + b_coef[i] * xdiff + c[i] * xdiff**2 + a_coef[i] * xdiff**3
            return yi
        return quad_spline

    # cubic spline
    elif order == 3:
        xdiff = np.diff(xd)  # how far apart the xd values are
        ydiff = np.diff(yd)  # the difference between yd values

        # now we set up the system of equations to find second derivatives c (continuity constraints)
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)

        for i in range(1, n):
            A[i, i - 1] = xdiff[i - 1]  # 2nd derivative at previous point
            A[i, i] = 2 * (xdiff[i - 1] + xdiff[i])  # 2nd derivative at current point
            A[i, i + 1] = xdiff[i]  # 2nd derivative at next point
            # ensures the slope at the points is continuous
            b[i] = 3 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])

        # clarify boundary conditions
        A[0, 0] = 1  # sets first 2nd derivative to 0
        A[-1, -1] = 1  # sets last 2nd derivative to 0

        # solve for c and compute coefficients
        c = np.linalg.solve(A, b)
        a_coef = (c[1:] - c[:-1]) / (3 * xdiff)
        b_coef = (ydiff / xdiff) - xdiff * (c[1:] + 2 * c[:-1]) / 3
        c_coef = yd[:-1]  # original y values for each segment

        # now we compute the spline
        def cubic_spline(xi):
            i = np.searchsorted(xd, xi) - 1
            i = np.clip(i, 0, n - 1)
            xdiff = xi - xd[i]
            yi = c_coef[i] + b_coef[i] * xdiff + c[i] * xdiff ** 2 + a_coef[i] * xdiff ** 3
            return yi
        return cubic_spline

    else:
        if order != [1, 2, 3]:
            raise ValueError('The order must be 1, 2, or 3.')


