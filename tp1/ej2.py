import numpy as np
import pandas as pd
import utils

sensor_positions_file = 'sensor_positions.txt'
measurements_file = 'measurements.txt'
trajectory_file = 'trajectory.txt'

sensor_data = pd.read_csv(sensor_positions_file)
sensors_data = sensor_data[['x_i(m)', 'y_i(m)', 'z_i(m)']].values

measurement_data = pd.read_csv(measurements_file)

trajectory_data = pd.read_csv(trajectory_file)


estimated_positions = []
for index, row in measurement_data.iterrows():
    measurements = np.array([row['d1(m)'], row['d2(m)'], row['d3(m)']])
    estimated_position = utils.newton_raphson(utils.f, sensors_data, measurements)
    estimated_positions.append([row['t(s)'], *estimated_position])

estimated_positions_df = pd.DataFrame(estimated_positions, columns=['t', 'x', 'y', 'z'])

spline_x = utils.interpolate_with_cubic_spline(estimated_positions_df['t'], estimated_positions_df['x'])
spline_y = utils.interpolate_with_cubic_spline(estimated_positions_df['t'], estimated_positions_df['y'])
spline_z = utils.interpolate_with_cubic_spline(estimated_positions_df['t'], estimated_positions_df['z'])

t_real = trajectory_data['t(s)']

x_interp = spline_x(t_real)
y_interp = spline_y(t_real)
z_interp = spline_z(t_real)

x_real = trajectory_data['x(m)']
y_real = trajectory_data['y(m)']
z_real = trajectory_data['z(m)']

utils.calculate_and_show_rmse(x_real, x_interp, y_real, y_interp, z_real, z_interp)

utils.plot_and_show_trajectories(t_real, x_real, x_interp, y_real, y_interp, z_real, z_interp)
