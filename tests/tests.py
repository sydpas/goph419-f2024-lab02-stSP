import numpy as np
import scipy as sp
from src.lab02.linalg_interp import (
    gauss_iter_solve,
    spline_function,
)


def main():
    # first we check gauss_iter_solve:
    print(f'Testing gauss_iter_solve...')

    A = np.array([[10, 1, 1],
                  [2, 10, 1],
                  [2, 1, 10]])
    b = np.array([12, 13, 14])
    x0 = np.zeros(len(b))

    # test 1: check that all x-producing algorithms match
    print(f'Test one...')

    x_gauss_seidel = gauss_iter_solve(A, b, x0, 1e-8, 'seidel')
    x_gauss_jacobi = gauss_iter_solve(A, b, x0, 1e-8, 'jacobi')
    x_np = np.linalg.solve(A, b)

    print(f'My seidel algorithm: {x_gauss_seidel}')
    print(f'My jacobi algorithm: {x_gauss_jacobi}')
    print(f'The NumPy algorithm: {x_np}')

    # test 2: check that AA^-1 = I
    print(f'Test two...')

    # we will use the same A as above but a different b and x0
    b = np.eye(A.shape[0])  # creates identity matrix with same rows and columns as A
    x0 = np.zeros_like(b)

    A_inverse_seidel = gauss_iter_solve(A, b, x0, 1e-8, 'seidel')
    A_inverse_jacobi = gauss_iter_solve(A, b, x0, 1e-8, 'jacobi')
    A_inverse = np.linalg.inv(A)

    print(f'My seidel algorithm: {A_inverse_seidel}')
    print(f'My jacobi algorithm: {A_inverse_jacobi}')
    print(f'The NumPy algorithm: {A_inverse}')

    con_number = np.linalg.cond(A)
    print(f'Condition number: {con_number}')

    # now we check spline_function:
    print(f'Testing spline_function...')

    # data i will use:
    x = [0, 0.9, 2.4, 3, 4.6, 5, 6, 7, 8.4, 9]
    y = [2, 9, 10, 20, 42, 57, 80, 93, 100, 137]

    # test 1: check if linear, quad, and cubic return correct data
    print(f'Test one...')

    # my functions
    linear_func = spline_function(x, y, 1)
    quad_func = spline_function(x, y, 2)
    cubic_func = spline_function(x, y, 3)

    # scipy functions
    linear_sp = sp.interpolate.interp1d(x, y, kind='linear')
    quad_sp = sp.interpolate.interp1d(x, y, kind='quadratic')
    cubic_sp = sp.interpolate.CubicSpline(x, y)

    # now retrieving the values
    x_test = np.linspace(0, 9, 5)

    # using my functions
    y_linear = [linear_func(xi) for xi in x_test]
    y_quad = [quad_func(xi) for xi in x_test]
    y_cubic = [cubic_func(xi) for xi in x_test]

    # using scipy splines
    y_linear_sp = linear_sp(x_test)
    y_quad_sp = quad_sp(x_test)
    y_cubic_sp = cubic_sp(x_test)

    print(f'My linear spline: {y_linear}')
    print(f'My quadratic spline: {y_quad}')
    print(f'My cubic spline: {y_cubic}')
    print(f'The SciPy linear spline: {y_linear_sp}')
    print(f'The SciPy quadratic spline: {y_quad_sp}')
    print(f'The SciPy cubic spline: {y_cubic_sp}')

    # test 2: comparing my cubic spline with the univariate scipy function
    print(f'Test two...')

    # i will use my function i made
    my_cubic_spline = spline_function(x, y, 3)
    # using the univariate function
    uni_spline = sp.interpolate.UnivariateSpline(x, y, s=0, ext='raise')

    # retrieving the values
    x_test = np.linspace(0, 9, 5)

    my_cubic_test = [my_cubic_spline(xi) for xi in x_test]
    uni_spline_test = uni_spline(x_test)

    print(f'My cubic spline: {my_cubic_test}')
    print(f'The Univariate Spline: {uni_spline_test}')


if __name__ == "__main__":
    main()