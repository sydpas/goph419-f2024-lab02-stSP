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

    # now we will plot all the points
    fig = plt.figure(figsize=(12, 8))  # width, height

    # linear plot for water
    plt.subplot(3, 2, 1)  # plot 1
    plt.plot(xd1, yd1, '-', color='blue', label='Data')
    plt.plot(x_linear_water, y_linear_water, 'o', color='green', markersize=2, label='Linear Spline')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Water Density (g/cm\u00b3)')
    plt.legend()
    plt.grid(True)

    # quad plot for water
    plt.subplot(3, 2, 3)  # plot 2
    plt.plot(xd1, yd1, '--', color='green', label='Data')
    plt.plot(x_quad_water, y_quad_water, 's', color='red', markersize=2, label='Quadratic Spline')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Water Density (g/cm\u00b3)')
    plt.legend()
    plt.grid(True)

    # cubic plot for water
    plt.subplot(3, 2, 5)  # plot 3
    plt.plot(xd1, yd1, '-.', color='red', label='Data')
    plt.plot(x_cubic_water, y_cubic_water, '^', color='blue', markersize=2, label='Cubic Spline')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Water Density (g/cm\u00b3)')
    plt.legend()
    plt.grid(True)

    # linear plot for air
    plt.subplot(3, 2, 2)  # 3 rows, 1 column, plot 1
    plt.plot(xd2, yd2, '-', color='cyan', label='Data')
    plt.plot(x_linear_air, y_linear_air, 'x', color='purple', markersize=2, label='Linear Spline')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Air Density (kg/m\u00b3)')
    plt.legend()
    plt.grid(True)

    # quad plot for air
    plt.subplot(3, 2, 4)  # 3 rows, 1 column, plot 2
    plt.plot(xd2, yd2, '--', color='purple', label='Data')
    plt.plot(x_quad_air, y_quad_air, 'p', color='hotpink', markersize=2, label='Quadratic Spline')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Air Density (kg/m\u00b3)')
    plt.legend()
    plt.grid(True)

    # cubic plot for air
    plt.subplot(3, 2, 6)  # 3 rows, 1 column, plot 3
    plt.plot(xd2, yd2, '-.', color='hotpink', label='Data')
    plt.plot(x_cubic_air, y_cubic_air, 'd', color='cyan', markersize=2, label='Cubic Spline')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Air Density (kg/m\u00b3)')
    plt.legend()
    plt.grid(True)

    fig.text(0.25, 0.95, 'Water Density vs. Temperature', fontsize=12, fontweight='bold', ha='center')
    fig.text(0.75, 0.95, 'Air Density vs. Temperature', fontsize=12, fontweight='bold', ha='center')

    plt.tight_layout(rect=(0, 0, 0.95, 0.95))  # leave room for the title
    plt.savefig('C:\\Users\\sydne\\git\\goph419\\goph419-f2024-lab02-stSP\\figures\\spline_functions.png')
    plt.show()


if __name__ == "__main__":
    main()