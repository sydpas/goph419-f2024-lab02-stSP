import numpy as np
import matplotlib.pyplot as plt
from src.lab02.linalg_interp import (
    spline_function,
)

w_d = np.loadtxt("../data/water_density_vs_temp_usgs.txt", "float")
a_d = np.loadtxt("../data/air_density_vs_temp_eng_toolbox.txt", "float")

xd1, yd1 = w_d[:, 0], w_d[:, 1]
xd2, yd2 = a_d[:, 0], a_d[:, 1]


def main():
    print(f'Using water density data...')

    linear_spline = spline_function(xd1, yd1, 1)
    x_linear_water = np.linspace(min(xd1), max(xd1), 100)  # creating 100 equally spaced values
    y_linear_water = [linear_spline(x) for x in x_linear_water]

    quad_spline = spline_function(xd1, yd1, 2)
    x_quad_water = np.linspace(min(xd1), max(xd1), 100)
    y_quad_water = [quad_spline(x) for x in x_quad_water]

    cubic_spline = spline_function(xd1, yd1, 3)
    x_cubic_water = np.linspace(min(xd1), max(xd1), 100)
    y_cubic_water = [cubic_spline(x) for x in x_cubic_water]

    print(f'Linear: {y_linear_water[:5]}...')  # Check for non-zero results
    print(f'Quadratic: {y_quad_water[:5]}...')
    print(f'Cubic: {y_cubic_water[:5]}...')

    print(f'Using air density data...')

    linear_spline = spline_function(xd2, yd2, 1)
    x_linear_air = np.linspace(min(xd2), max(xd2), 100)  # creating 100 equally spaced values
    y_linear_air = [linear_spline(x) for x in x_linear_air]

    quad_spline = spline_function(xd2, yd2, 2)
    x_quad_air = np.linspace(min(xd2), max(xd2), 100)
    y_quad_air = [quad_spline(x) for x in x_quad_air]

    cubic_spline = spline_function(xd2, yd2, 3)
    x_cubic_air = np.linspace(min(xd2), max(xd2), 100)
    y_cubic_air = [cubic_spline(x) for x in x_cubic_air]

    print(f'Linear: {y_linear_air[:5]}...')  # Check for non-zero results
    print(f'Quadratic: {y_quad_air[:5]}...')
    print(f'Cubic: {y_cubic_air[:5]}...')

    # now we will plot all the points

    fig = plt.figure(figsize=(8, 6))

    # linear plot for water
    plt.subplot(3, 2, 1)  # plot 1
    plt.plot(x_linear_water, y_linear_water, label='Linear Spline', color='darkblue')
    plt.title('Linear Spline', fontsize=14, fontweight='bold', color='darkblue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    # quad plot for water
    plt.subplot(3, 2, 3)  # plot 2
    plt.plot(x_linear_air, y_cubic_air, label='Cubic Spline', color='darkgreen')
    plt.title('Quadratic Spline', fontsize=14, fontweight='bold', color='darkgreen')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    # cubic plot for water
    plt.subplot(3, 2, 5)  # plot 3
    plt.plot(x_linear_water, y_cubic_water, label='Cubic Spline', color='darkred')
    plt.title('Cubic Spline', fontsize=14, fontweight='bold', color='darkred')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    # linear plot for air
    plt.subplot(3, 2, 2)  # 3 rows, 1 column, plot 1
    plt.plot(x_linear_water, y_linear_water, label='Linear Spline', color='darkblue')
    plt.title('Linear Spline', fontsize=14, fontweight='bold', color='darkblue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    # quad plot for air
    plt.subplot(3, 2, 4)  # 3 rows, 1 column, plot 2
    plt.plot(x_linear_air, y_cubic_air, label='Cubic Spline', color='darkgreen')
    plt.title('Quadratic Spline', fontsize=14, fontweight='bold', color='darkgreen')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    # cubic plot for air
    plt.subplot(3, 2, 6)  # 3 rows, 1 column, plot 3
    plt.plot(x_linear_air, y_cubic_air, label='Cubic Spline', color='darkred')
    plt.title('Cubic Spline', fontsize=14, fontweight='bold', color='darkred')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    fig.text(0.28, 0.95, 'Water Density vs. Temperature', fontsize=12, fontweight='bold', ha='center')
    fig.text(0.75, 0.95, 'Air Density vs. Temperature', fontsize=12, fontweight='bold', ha='center')

    plt.tight_layout(rect=(0, 0, 0.92, 0.92))  # leave room for the title
    plt.savefig('C:\\Users\\sydne\\git\\goph419\\goph419-f2024-lab02-stSP\\figures\\spline_functions.png')
    plt.show()


if __name__ == "__main__":
    main()