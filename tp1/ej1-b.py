import numpy as np
import utils


x_for_real_values = np.linspace(-1, 1, 100)
y_for_real_values = np.linspace(-1, 1, 100)
X_real, Y_real, Z_real = utils.get_x_y_z_values(x_for_real_values, y_for_real_values, utils.f2)

utils.plot_3d_figure(X_real, Y_real, Z_real)

number_of_points = (3, 10, 20)

for n in number_of_points:
    utils.interpolate_and_plot(n, utils.f2, 'linear')
    utils.interpolate_and_plot(n, utils.f2, 'cubic')

number_of_points = 20
error_linear = utils.calculate_error(number_of_points, utils.f2, 'linear')[0]
error_cubic = utils.calculate_error(number_of_points, utils.f2, 'cubic')[0]
error_nearest = utils.calculate_error(number_of_points, utils.f2, 'nearest')[0]

utils.plot_errors(range(5, number_of_points + 1), linear=error_linear, cubic=error_cubic,
            nearest=error_nearest)

error_linear_max = utils.calculate_error(number_of_points, utils.f2, 'linear')[1]
error_cubic_max = utils.calculate_error(number_of_points, utils.f2, 'cubic')[1]
error_nearest_max = utils.calculate_error(number_of_points, utils.f2, 'nearest')[1]

utils.plot_errors(range(5, number_of_points + 1), linear=error_linear_max,
            cubic=error_cubic_max,
            nearest=error_nearest_max)

error_linear_chebyshev = utils.calculate_error_chebyshev(number_of_points, utils.f2, 'linear')[0]
error_cubic_chebyshev = utils.calculate_error_chebyshev(number_of_points, utils.f2, 'cubic')[0]
error_nearest_chebyshev = utils.calculate_error_chebyshev(number_of_points, utils.f2, 'nearest')[0]

utils.plot_errors(range(5, number_of_points + 1), linear_chebyshev=error_linear_chebyshev,
            cubic_chebyshev=error_cubic_chebyshev, nearest_chebyshev=error_nearest_chebyshev)

error_linear_chebyshev_max = utils.calculate_error_chebyshev(number_of_points, utils.f2, 'linear')[1]
error_cubic_chebyshev_max = utils.calculate_error_chebyshev(number_of_points, utils.f2, 'cubic')[1]
error_nearest_chebyshev_max = utils.calculate_error_chebyshev(number_of_points, utils.f2, 'nearest')[1]

utils.plot_errors(range(5, number_of_points + 1),
            linear_chebyshev=error_linear_chebyshev_max,
            cubic_chebyshev=error_cubic_chebyshev_max, nearest_chebyshev=error_nearest_chebyshev_max)
