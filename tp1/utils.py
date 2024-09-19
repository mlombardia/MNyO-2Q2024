import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.interpolate import lagrange, CubicSpline
import math
from scipy.interpolate import griddata

n_global = 100


def f1(x):
    return -0.4 * math.tanh(50 * x) + 0.6


def f2(x1, x2):
    t1 = 0.75 * (np.exp(((-(9 * x1 - 2) ** 2) / 4) - (((9 * x2 - 2) ** 2) / 4)))
    t2 = 0.75 * (np.exp(((-(9 * x1 + 1) ** 2) / 49) - (((9 * x2 + 1) ** 2) / 10)))
    t3 = 0.5 * (np.exp(((-(9 * x1 - 7) ** 2) / 4) - (((9 * x2 - 3) ** 2) / 4)))
    t4 = - 0.2 * (np.exp(((-(9 * x1 - 7) ** 2) / 4) - (((9 * x2 - 3) ** 2) / 4)))
    return t1 + t2 + t3 + t4


def get_y_values_of_function(x_values, f):
    return np.array([f(x) for x in x_values])


def get_equidistant_points(start, end, n):
    return np.linspace(start, end, n)


def get_chebyshev_points(start, end, n):
    return np.array(
        [0.5 * (start + end) + 0.5 * (end - start) * math.cos((2 * i + 1) * math.pi / (2 * n)) for i in range(n)])


def interpolate_with_lagrange(x_values, y_values):
    return lagrange(x_values, y_values)


x_100_points = get_equidistant_points(-1, 1, n_global)
y_100_points = get_y_values_of_function(x_100_points, f1)


def calculate_and_plot_lagrange_interpolation(f, points, start, end):
    for i in points:
        x_values = get_equidistant_points(start, end, i)
        y_values = get_y_values_of_function(x_values, f)

        lagrange_interpolation = interpolate_with_lagrange(x_values, y_values)

        x_values_to_plot = get_equidistant_points(start, end, n_global)
        y_values_to_plot = lagrange_interpolation(x_values_to_plot)

        plot_multiple_datasets((np.column_stack((x_values_to_plot, y_values_to_plot)),
                                'Interpolacion de Lagrange con ' + str(i) + ' puntos'),
                               (np.column_stack((x_100_points, y_100_points)), 'Funcion original'))


def calculate_and_plot_cubic_spline_interpolation(f, points, start, end):
    for i in points:
        x_values = get_equidistant_points(start, end, i)
        y_values = get_y_values_of_function(x_values, f)

        cubic_spline_interpolation = interpolate_with_cubic_spline(x_values, y_values)

        x_values_to_plot = get_equidistant_points(start, end, n_global)
        y_values_to_plot = cubic_spline_interpolation(x_values_to_plot)

        plot_multiple_datasets((np.column_stack((x_values_to_plot, y_values_to_plot)),
                                'Interpolacion de Spline Cubico con ' + str(i) + ' puntos'),
                               (np.column_stack((x_100_points, y_100_points)), 'Funcion original'))


def calculate_and_plot_lagrange_interpolation_chebyshev(f, points, start, end):
    for i in points:
        x_values = get_chebyshev_points(start, end, i)
        x_values = np.sort(x_values)
        y_values = get_y_values_of_function(x_values, f)

        lagrange_interpolation = interpolate_with_lagrange(x_values, y_values)

        x_values_to_plot = get_equidistant_points(start, end, n_global)
        y_values_to_plot = lagrange_interpolation(x_values_to_plot)

        plot_multiple_datasets((np.column_stack((x_values_to_plot, y_values_to_plot)),
                                'Interpolacion de Lagrange con ' + str(i) + ' puntos (cheb)'),
                               (np.column_stack((x_100_points, y_100_points)), 'Funcion original'))


def calculate_and_plot_cubic_spline_interpolation_chebyshev(f, points, start, end):
    for i in points:
        x_values = get_chebyshev_points(start, end, i)
        x_values = np.sort(x_values)
        y_values = get_y_values_of_function(x_values, f)

        cubic_spline_interpolation = interpolate_with_cubic_spline(x_values, y_values)

        x_values_to_plot = get_equidistant_points(start, end, n_global)
        y_values_to_plot = cubic_spline_interpolation(x_values_to_plot)

        plot_multiple_datasets((np.column_stack((x_values_to_plot, y_values_to_plot)),
                                'Interpolacion de Spline Cubico con ' + str(i) + ' puntos (cheb)'),
                               (np.column_stack((x_100_points, y_100_points)), 'Funcion original'))


def calculate_error_lagrange(f, start, end, n):
    average_error_interpolation_lagrange = []

    x_values_to_original = get_equidistant_points(start, end, n_global)
    y_values_original = get_y_values_of_function(x_values_to_original, f)

    for i in range(4, n):
        x_values = get_equidistant_points(start, end, i)
        y_values = get_y_values_of_function(x_values, f)

        interpolation_lagrange = interpolate_with_lagrange(x_values, y_values)
        y_values_to_plot_lagrange = interpolation_lagrange(x_values_to_original)

        average_error_n = np.mean(np.abs(y_values_original - y_values_to_plot_lagrange))
        average_error_interpolation_lagrange.append(average_error_n)

    return average_error_interpolation_lagrange


def calculate_error_cubic_spline(f, start, end, n):
    average_error_interpolation_cubic_spline = []

    x_values_to_original = get_equidistant_points(start, end, n_global)
    y_values_original = get_y_values_of_function(x_values_to_original, f)

    for i in range(4, n):
        x_values = get_equidistant_points(start, end, i)
        y_values = get_y_values_of_function(x_values, f)

        interpolation_cubic_spline = interpolate_with_cubic_spline(x_values, y_values)
        y_values_to_plot_cubic_spline = interpolation_cubic_spline(x_values_to_original)

        average_error_n = np.mean(np.abs(y_values_original - y_values_to_plot_cubic_spline))
        average_error_interpolation_cubic_spline.append(average_error_n)

    return average_error_interpolation_cubic_spline


def calculate_error_lagrange_chebyshev(f, start, end, n):
    average_error_interpolation_lagrange = []

    x_values_to_original = get_equidistant_points(start, end, n_global)
    y_values_original = get_y_values_of_function(x_values_to_original, f)

    for i in range(4, n):
        x_values = get_chebyshev_points(start, end, i)
        x_values = np.sort(x_values)
        y_values = get_y_values_of_function(x_values, f)

        interpolation_lagrange = interpolate_with_lagrange(x_values, y_values)
        y_values_to_plot_lagrange = interpolation_lagrange(x_values_to_original)

        average_error_n = np.mean(np.abs(y_values_original - y_values_to_plot_lagrange))
        average_error_interpolation_lagrange.append(average_error_n)

    return average_error_interpolation_lagrange


def calculate_error_cubic_spline_chebyshev(f, start, end, n):
    average_error_interpolation_cubic_spline = []

    x_values_to_original = get_equidistant_points(start, end, n_global)
    y_values_original = get_y_values_of_function(x_values_to_original, f)

    for i in range(4, n):
        x_values = get_chebyshev_points(start, end, i)
        x_values = np.sort(x_values)
        y_values = get_y_values_of_function(x_values, f)

        interpolation_cubic_spline = interpolate_with_cubic_spline(x_values, y_values)
        y_values_to_plot_cubic_spline = interpolation_cubic_spline(x_values_to_original)

        average_error_n = np.mean(np.abs(y_values_original - y_values_to_plot_cubic_spline))
        average_error_interpolation_cubic_spline.append(average_error_n)

    return average_error_interpolation_cubic_spline

def calculate_error(n, f2, method):
    real_values_x_values = np.linspace(-1, 1, 100)
    real_values_y_values = np.linspace(-1, 1, 100)

    X_real, Y_real, Z_real = get_x_y_z_values(real_values_x_values, real_values_y_values, f2)

    errors = []
    errors_max = []

    for i in range(5, n + 1):  # Start from 2 to ensure at least 4 points
        x_values = np.linspace(-1, 1, i)
        y_values = np.linspace(-1, 1, i)

        X, Y, Z = get_x_y_z_values(x_values, y_values, f2)

        points = np.vstack((X.flatten(), Y.flatten())).T
        values = Z.flatten()

        grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]

        grid_z = griddata(points, values, (grid_x, grid_y), method=method)

        error = np.mean(np.abs(grid_z - Z_real))
        errors.append(error)

        error_max = np.max(np.abs(grid_z - Z_real))
        errors_max.append(error_max)

    return errors, errors_max


def calculate_error_chebyshev(n, f2, method):
    real_values_x_values = np.linspace(-1, 1, 100)
    real_values_y_values = np.linspace(-1, 1, 100)

    X_real, Y_real, Z_real = get_x_y_z_values(real_values_x_values, real_values_y_values, f2)

    errors = []
    errors_max = []

    for i in range(5, n + 1):  # Start from 2 to ensure at least 4 points
        x_values = np.cos(np.linspace(-np.pi, 0, i))
        y_values = np.cos(np.linspace(-np.pi, 0, i))

        X, Y, Z = get_x_y_z_values(x_values, y_values, f2)

        points = np.vstack((X.flatten(), Y.flatten())).T
        values = Z.flatten()

        grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]

        grid_z = griddata(points, values, (grid_x, grid_y), method=method)

        error = np.mean(np.abs(grid_z - Z_real))
        errors.append(error)

        error_max = np.max(np.abs(grid_z - Z_real))
        errors_max.append(error_max)

    return errors, errors_max
def plot_errors(n_values, **kwargs):
    plt.figure(figsize=(10, 6))
    for label, errors in kwargs.items():
        plt.plot(n_values, errors, marker='o', linestyle='-', label=label)

    # Set the locations of the x-ticks
    plt.xticks(n_values)

    # Get the minimum and maximum error values
    min_error = min(min(errors) for errors in kwargs.values())
    max_error = max(max(errors) for errors in kwargs.values())

    # Set the locations of the y-ticks
    plt.yticks(np.linspace(min_error, max_error, num=10))

    plt.xlabel('n puntos')
    plt.ylabel('error')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_errors_semilog(n_values, xlim=None, ylim=None, **kwargs):
    plt.figure(figsize=(10, 6))
    for label, errors in kwargs.items():
        plt.semilogy(n_values, errors, marker='o', linestyle='-', label=label)

    plt.xticks(n_values)

    min_error = min(min(errors) for errors in kwargs.values())
    max_error = max(max(errors) for errors in kwargs.values())

    plt.yticks(np.linspace(min_error, max_error, num=10))

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel('n puntos', fontsize=18)
    plt.ylabel('error', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_multiple_datasets(*args, xlim=None, ylim=None, **kwargs):
    plt.figure(figsize=(12, 8))

    for data in args:
        plt.plot(data[0][:, 0], data[0][:, 1], marker='o', linewidth=2,
                 label=data[1])
        plt.scatter(data[0][:, 0], data[0][:, 1])

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel('X coord (m)', fontsize=18)
    plt.ylabel('Y coord (m)', fontsize=18)
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()


def interpolate_with_cubic_spline(x_values, y_values):
    return CubicSpline(x_values, y_values)


def get_x_y_z_values(x_values, y_values, function):
    X, Y = np.meshgrid(x_values, y_values)
    Z = function(X, Y)
    return X, Y, Z


def plot_3d_figure(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_3d_figure_with_points(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c='r', marker='o', s=0.5)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def interpolate_and_plot(n, f2, method):
    x_values = np.linspace(-1, 1, n)
    y_values = np.linspace(-1, 1, n)

    X, Y, Z = get_x_y_z_values(x_values, y_values, f2)

    points = np.vstack((X.flatten(), Y.flatten())).T
    values = Z.flatten()

    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]

    grid_z = griddata(points, values, (grid_x, grid_y), method=method)
    print(grid_z)

    plot_3d_figure(grid_x, grid_y, grid_z)


def interpolate_and_plot_with_chebyshev_nodes(n, f2, method):
    x_values = np.cos(np.linspace(-np.pi, 0, n))
    y_values = np.cos(np.linspace(-np.pi, 0, n))

    X, Y, Z = get_x_y_z_values(x_values, y_values, f2)

    points = np.vstack((X.flatten(), Y.flatten())).T
    values = Z.flatten()

    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]

    grid_z = griddata(points, values, (grid_x, grid_y), method=method)

    plot_3d_figure(grid_x, grid_y, grid_z)


def f(x, sensors, distances):
    return np.array([
        np.sqrt((x[0] - sensors[0, 0]) ** 2 + (x[1] - sensors[0, 1]) ** 2 + (x[2] - sensors[0, 2]) ** 2) - distances[0],
        np.sqrt((x[0] - sensors[1, 0]) ** 2 + (x[1] - sensors[1, 1]) ** 2 + (x[2] - sensors[1, 2]) ** 2) - distances[1],
        np.sqrt((x[0] - sensors[2, 0]) ** 2 + (x[1] - sensors[2, 1]) ** 2 + (x[2] - sensors[2, 2]) ** 2) - distances[2]
    ])


def jacobian_matrix(x, sensors):
    matrix = np.zeros((3, 3))
    for i in range(3):
        dist = np.sqrt((x[0] - sensors[i, 0]) ** 2 + (x[1] - sensors[i, 1]) ** 2 + (x[2] - sensors[i, 2]) ** 2)
        if dist != 0:
            matrix[i, 0] = (x[0] - sensors[i, 0]) / dist
            matrix[i, 1] = (x[1] - sensors[i, 1]) / dist
            matrix[i, 2] = (x[2] - sensors[i, 2]) / dist
    return matrix


def newton_raphson(f, sensors, distances, initial=None, tol=1e-6, max_iter=10):
    if initial is None:
        initial = [0.0, 0.0, 0.0]
    x = np.array(initial)

    for _ in range(max_iter):
        evaluated_points = f(x, sensors, distances)
        evaluated_jacobian = jacobian_matrix(x, sensors)

        if np.linalg.norm(evaluated_points) < tol:
            break

        j_inv = np.linalg.inv(evaluated_jacobian)
        p = x - j_inv @ evaluated_points

        if np.linalg.norm(x - p) < tol:
            break

        x = p
    return x


def plot_and_show_trajectories(t, x, x_int, y, y_int, z, z_int):
    plt.figure(figsize=(6, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, x, '-', label='x real', linewidth=7)
    plt.plot(t, x_int, '-', label='x interpolada', linewidth=2)
    plt.xlabel('Tiempo (s)', fontsize=18)
    plt.ylabel('x (m)', fontsize=18)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, y, '-', label='y real', linewidth=7)
    plt.plot(t, y_int, '-', label='y interpolada', linewidth=2)
    plt.xlabel('Tiempo (s)', fontsize=18)
    plt.ylabel('y (m)', fontsize=18)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, z, '-', label='z real', linewidth=7)
    plt.plot(t, z_int, '-', label='z interpolada', linewidth=2)
    plt.xlabel('Tiempo (s)', fontsize=18)
    plt.ylabel('z (m)', fontsize=18)
    plt.legend()

    plt.tight_layout()
    plt.show()


def calculate_and_show_rmse(x, x_prime, y, y_prime, z, z_prime):
    rmse_x = np.sqrt(mean_squared_error(x, x_prime))
    rmse_y = np.sqrt(mean_squared_error(y, y_prime))
    rmse_z = np.sqrt(mean_squared_error(z, z_prime))

    print(f"RMSE en x: {rmse_x}")
    print(f"RMSE en y: {rmse_y}")
    print(f"RMSE en z: {rmse_z}")
