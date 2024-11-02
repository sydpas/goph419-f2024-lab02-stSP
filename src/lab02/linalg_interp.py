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

# value errors in the first main paragraph under the parameters

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
