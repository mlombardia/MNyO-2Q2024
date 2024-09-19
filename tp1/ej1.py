import utils

points_to_analyze = [10,20]

utils.calculate_and_plot_lagrange_interpolation(utils.f1, points_to_analyze, -1, 1)
utils.calculate_and_plot_cubic_spline_interpolation(utils.f1, points_to_analyze, -1, 1)

utils.calculate_and_plot_lagrange_interpolation_chebyshev(utils.f1, points_to_analyze, -1, 1)
utils.calculate_and_plot_cubic_spline_interpolation_chebyshev(utils.f1, points_to_analyze, -1, 1)

n_points_for_error = 35

error_lagrange = utils.calculate_error_lagrange(utils.f1, -1, 1, n_points_for_error)
error_cubic_spline = utils.calculate_error_cubic_spline(utils.f1, -1, 1, n_points_for_error)
error_lagrange_chebyshev = utils.calculate_error_lagrange_chebyshev(utils.f1, -1, 1, n_points_for_error)
error_cubic_spline_chebyshev = utils.calculate_error_cubic_spline_chebyshev(utils.f1, -1, 1, n_points_for_error)

n_values = [i for i in range(4, n_points_for_error)]

utils.plot_errors_semilog(n_values, xlim=[4,27], ylim=[0,100], Lagrange=error_lagrange, Cubic_Spline=error_cubic_spline,
            Lagrange_Chebyshev=error_lagrange_chebyshev, Cubic_Spline_Chebyshev=error_cubic_spline_chebyshev)

utils.plot_errors_semilog(n_values, xlim=[4,25], ylim=[0,0.1], Cubic_Spline=error_cubic_spline,
            Lagrange_Chebyshev=error_lagrange_chebyshev, Cubic_Spline_Chebyshev=error_cubic_spline_chebyshev)
