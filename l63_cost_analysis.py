"""
Run the algorithm on L63 for some time
Record the total cost and cost per unit time
"""

import numpy as np
from FixedGPR import RBF, MultiGPR
from sklearn.neighbors import KernelDensity
from odelibrary import L63
import matplotlib.pyplot as plt
from numpy.linalg import norm
from itertools import product
import time


def rk4(x, f, dt):
    k1 = f(x)
    k2 = f(x + dt * k1 / 2)
    k3 = f(x + dt * k2 / 2)
    k4 = f(x + dt * k3)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def cov_norm(cov):
    return np.max(cov)

initial = np.array([1, 1, 1])
inv_measure_initial = np.random.normal(np.array([1, 1, 1]), np.array([4, 4, 4]))
system = L63()
dt = 0.05
total_time = 40
inv_measure_time = 1000

kernel = RBF(1.0, 20.)
gpr = MultiGPR(3, kernel, alpha=1e-10)
cov_threshold = 1e-7

true_cost = 3
surrogate_cost = 1

if __name__ == "__main__":
    # total cost, number of training points, timeseries, surrogate error
    fig1, ((timeseries_ax, training_pts_ax), (total_cost_ax, error_ax)) = plt.subplots(2, 2)
    timeseries_ax.set_title("Timeseries")
    training_pts_ax.set_title("Number of training points")
    total_cost_ax.set_title("Total cost")
    error_ax.set_title("Surrogate error")

    plt.show()
    plt.ion()

    # Initialize data
    steps = int(total_time / dt)
    cost_data = np.arange(0, total_time, dt)
    total_cost = np.zeros(steps)
    alt_cost = np.zeros(steps)
    training_pts = np.zeros(steps)
    error = []
    error_data = []
    error_max = 0
    cost_max = 0
    # Begin plot
    surrogate_cost_line, = total_cost_ax.plot(cost_data, total_cost)
    alt_cost_line, = total_cost_ax.plot(cost_data, alt_cost)
    training_pts_line, = training_pts_ax.plot(cost_data, training_pts)
    error_line, = error_ax.plot([0], [0])

    # Run simulation
    position = initial
    for step in range(steps):
        # Try surrogate
        next_candidate = gpr.predict([position])
        # Parse surrogate output
        next_mean = np.array([coord.mean[0] for coord in next_candidate]).flatten()
        next_cov = cov_norm([coord.cov[0][0] for coord in next_candidate])
        # Update data
        total_cost[step:] += surrogate_cost
        alt_cost[step:] += true_cost

        true_next = rk4(position, lambda x: system.rhs(x, 0), dt)
        if next_cov > cov_threshold:
            gpr.add_fit(position, true_next)
            position = true_next
            total_cost[step:] += true_cost
            training_pts[step:] += 1

            # Update plots
            timeseries_ax.scatter(step * dt, position[0], color="yellow")
        else:
            error_data.append(step * dt)
            error.append(norm(next_mean - true_next))
            error_max = max(error_max, error[-1])
            position = next_mean

            # Update plots
            timeseries_ax.scatter(step * dt, position[0], color="blue")

        # Update limits
        cost_max = max(total_cost[-1], alt_cost[-1])

        surrogate_cost_line.set_ydata(total_cost)
        alt_cost_line.set_ydata(alt_cost)
        training_pts_line.set_ydata(training_pts)
        error_line.set_data(error_data, error)

        error_ax.set_xlim(0, dt * step)
        total_cost_ax.set_xlim(0, dt * step)
        training_pts_ax.set_xlim(0, dt * step)

        total_cost_ax.set_ylim(0, cost_max)
        error_ax.set_ylim(0, error_max)
        training_pts_ax.set_ylim(0, training_pts[-1])

        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.01)

    # Get timeseries of resulting GPR and L63
    timeseries_steps = int(inv_measure_time / dt) + 1
    l63_timeseries = np.empty((timeseries_steps, 3))
    gpr_timeseries = np.empty((timeseries_steps, 3))
    l63_timeseries[0, ] = inv_measure_initial
    gpr_timeseries[0, ] = inv_measure_initial
    for step in range(1, timeseries_steps):
        l63_timeseries[step, ] = rk4(l63_timeseries[step - 1, ], lambda x: system.rhs(x, 0), dt)
        # Get next GPR prediction
        next_candidate = gpr.predict([gpr_timeseries[step - 1, ]])
        gpr_timeseries[step, ] = np.array([coord.mean[0] for coord in next_candidate]).flatten()

    # Estimate invariant measure
    # l63_kde = KernelDensity(bandwidth=1)
    # l63_kde.fit(l63_timeseries)
    # gpr_kde = KernelDensity(bandwidth=1)
    # gpr_kde.fit(gpr_timeseries)

    l63_lists = l63_timeseries.transpose()
    gpr_lists = gpr_timeseries.transpose()

    l63_x_kde = KernelDensity()
    l63_x_kde.fit(l63_lists[0].reshape(-1, 1))
    l63_y_kde = KernelDensity()
    l63_y_kde.fit(l63_lists[1].reshape(-1, 1))
    l63_z_kde = KernelDensity()
    l63_z_kde.fit(l63_lists[2].reshape(-1, 1))

    gpr_x_kde = KernelDensity()
    gpr_x_kde.fit(gpr_lists[0].reshape(-1, 1))
    gpr_y_kde = KernelDensity()
    gpr_y_kde.fit(gpr_lists[1].reshape(-1, 1))
    gpr_z_kde = KernelDensity()
    gpr_z_kde.fit(gpr_lists[2].reshape(-1, 1))

    grid_points = 100
    x_inputs = np.linspace(-30, 30, grid_points)
    y_inputs = np.linspace(-30, 30, grid_points)
    z_inputs = np.linspace(-20, 60, grid_points)

    # Plot invariant measures
    fig2, ((true_x_inv_ax, true_y_inv_ax, true_z_inv_ax)) = plt.subplots(1, 3)

    true_x_inv_ax.plot(x_inputs, l63_x_kde.score_samples(x_inputs.reshape(-1, 1)), label="l63")
    true_x_inv_ax.plot(x_inputs, gpr_x_kde.score_samples(x_inputs.reshape(-1, 1)), label="gpr")
    true_x_inv_ax.set_title("x")
    true_x_inv_ax.legend()
    true_y_inv_ax.plot(y_inputs, l63_y_kde.score_samples(y_inputs.reshape(-1, 1)), label="l63")
    true_y_inv_ax.plot(y_inputs, gpr_y_kde.score_samples(y_inputs.reshape(-1, 1)), label="gpr")
    true_y_inv_ax.set_title("y")
    true_y_inv_ax.legend()
    true_z_inv_ax.plot(z_inputs, l63_z_kde.score_samples(z_inputs.reshape(-1, 1)), label="l63")
    true_z_inv_ax.plot(z_inputs, gpr_z_kde.score_samples(z_inputs.reshape(-1, 1)), label="gpr")
    true_z_inv_ax.set_title("z")
    true_z_inv_ax.legend()

    # true_x_inputs = np.array(list(product(x_inputs, [0], [20])))
    # true_x_inv_ax.plot(x_inputs, l63_kde.score_samples(true_x_inputs), label="l63")
    # true_x_inv_ax.plot(x_inputs, gpr_kde.score_samples(true_x_inputs), label="gpr")
    # true_x_inv_ax.set_title("x")
    # true_x_inv_ax.legend()
    # true_y_inputs = np.array(list(product([0], y_inputs, [20])))
    # true_y_inv_ax.plot(y_inputs, l63_kde.score_samples(true_y_inputs), label="l63")
    # true_y_inv_ax.plot(y_inputs, gpr_kde.score_samples(true_y_inputs), label="gpr")
    # true_y_inv_ax.set_title("y")
    # true_y_inv_ax.legend()
    # true_z_inputs = np.array(list(product([0], [0], z_inputs)))
    # true_z_inv_ax.plot(z_inputs, l63_kde.score_samples(true_z_inputs), label="l63")
    # true_z_inv_ax.plot(z_inputs, gpr_kde.score_samples(true_z_inputs), label="gpr")
    # true_z_inv_ax.set_title("z")
    # true_z_inv_ax.legend()
    plt.show()