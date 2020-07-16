from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from odelibrary import L63
from forward_euler import Forward_Euler
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gauss
import time

dt = 0.01
system = L63()
initial = np.array([1, 1, 1])
training_initial = np.array([1, 1.1, 1])
training_quantity = 5000
training_start = 0
training_end = 1000
surrogate_learning_threshold = [0.1, 0.1, 0.1]
start_time = 0
end_time = 20
spin_up = 20

relearning = True
x_threshold = 0.1
y_threshold = 0.1
z_threshold = 0.1

if __name__ == "__main__":
    # pre_gpr_1 = gauss.GaussianProcessRegressor(
    #     kernel=ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e6)) + WhiteKernel(1.0, (1e-10, 1e6)),
    #     n_restarts_optimizer=5)
    # pre_gpr_2 = gauss.GaussianProcessRegressor(
    #     kernel=ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e6)) + WhiteKernel(1.0, (1e-10, 1e6)),
    #     n_restarts_optimizer=5)
    # pre_gpr_3 = gauss.GaussianProcessRegressor(
    #     kernel=ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e6)) + WhiteKernel(1.0, (1e-10, 1e6)),
    #     n_restarts_optimizer=5)
    #
    #
    ## Gather initial training data
    training_timeseries = integrate.solve_ivp(lambda t, S: system.rhs(S, t), (training_start, training_end), training_initial, Forward_Euler)
    training_indices = np.random.randint(0, len(training_timeseries) - 1, training_quantity)
    data = np.transpose(training_timeseries.y)
    training_x = np.take(data, training_indices, axis=0)
    training_y = np.take(data, training_indices + 1, axis=0)
    #
    # print("Starting fit")
    # pre_gpr_1.fit(training_x, training_y[:, 0])
    # print("Kernel x", pre_gpr_1.kernel_)
    # pre_gpr_2.fit(training_x, training_y[:, 1])
    # print("Kernel y", pre_gpr_2.kernel_)
    # pre_gpr_3.fit(training_x, training_y[:, 2])
    # print("Kernel z", pre_gpr_3.kernel_)

    # Fix kernels
    # kernel_x = 15.3**2 * RBF(length_scale=36.5) + WhiteKernel(1e-10)
    # kernel_y = 15.3**2 * RBF(length_scale=36.5) + WhiteKernel(1e-10)
    # kernel_z = 9.55**2 * RBF(length_scale=28.3) + WhiteKernel(1e-10)
    kernel_x = 10**2 * RBF(length_scale=25)
    kernel_y = 10**2 * RBF(length_scale=25)
    kernel_z = 10**2 * RBF(length_scale=25)

    gpr_1 = gauss.GaussianProcessRegressor(kernel=kernel_x, optimizer=None)
    gpr_2 = gauss.GaussianProcessRegressor(kernel=kernel_y, optimizer=None)
    gpr_3 = gauss.GaussianProcessRegressor(kernel=kernel_z, optimizer=None)

    training_x1 = training_x
    training_x2 = training_x
    training_x3 = training_x
    training_y1 = training_y[:, 0]
    training_y2 = training_y[:, 1]
    training_y3 = training_y[:, 2]

    print("Starting GPR fit")
    gpr_1.fit(training_x1, training_y1)
    print("Fit x")
    gpr_2.fit(training_x2, training_y2)
    print("Fit y")
    gpr_3.fit(training_x3, training_y3)
    print("Fit z")

    ## Gather truth
    truth = integrate.solve_ivp(lambda t, S: system.rhs(S, t), (start_time, end_time), initial, Forward_Euler)
    surrogate_prediction = np.empty((len(truth.t), 3))
    surrogate_prediction[0] = initial

    true_model_points1 = np.empty((0, 3))
    true_model_points2 = np.empty((0, 3))
    true_model_points3 = np.empty((0, 3))

    ## Run model
    reversion_density = np.zeros((3, len(truth.t))) # Approximately the number of number of times reverting to the true model per time unit
    for i in range(1, len(surrogate_prediction)):
        previous = surrogate_prediction[i - 1].reshape(1, -1)
        (x, x_std) = gpr_1.predict(previous, return_std=True)
        (y, y_std) = gpr_2.predict(previous, return_std=True)
        (z, z_std) = gpr_3.predict(previous, return_std=True)
        x = x[0]
        y = y[0]
        z = z[0]
        x_std = x_std[0]
        y_std = y_std[0]
        z_std = z_std[0]

        true_next = surrogate_prediction[i - 1] + dt * system.rhs(surrogate_prediction[i - 1], i * dt)

        if x_std > x_threshold:
            x = true_next[0]
            print("Reverted to true model for x at time", i * dt, "with confidence", x_std)
            true_model_points1 = np.append(true_model_points1, [surrogate_prediction[i - 1]], axis=0)
            for j in range(max(i - 50, 0), min(i + 51, len(surrogate_prediction))):
                reversion_density[0, j] += 1
            if relearning:
                training_x1 = np.append(training_x1, [surrogate_prediction[i - 1]], axis=0)
                training_y1 = np.append(training_y1, [true_next[0]])
                gpr_1.fit(training_x1, training_y1)
        if y_std > y_threshold:
            y = true_next[1]
            print("Reverted to true model for y at time", i * dt, "with confidence", y_std)
            true_model_points2 = np.append(true_model_points1, [surrogate_prediction[i - 1]], axis=0)
            for j in range(max(i - 50, 0), min(i + 51, len(surrogate_prediction))):
                reversion_density[1, j] += 1
            if relearning:
                training_x2 = np.append(training_x2, [surrogate_prediction[i - 1]], axis=0)
                training_y2 = np.append(training_y2, [true_next[1]])
                gpr_2.fit(training_x2, training_y2)
        if z_std > z_threshold:
            z = true_next[2]
            print("Reverted to true model for z at time", i * dt, "with confidence", z_std)
            true_model_points3 = np.append(true_model_points1, [surrogate_prediction[i - 1]], axis=0)
            for j in range(max(i - 50, 0), min(i + 51, len(surrogate_prediction))):
                reversion_density[2, j] += 1
            if relearning:
                training_x3 = np.append(training_x3, [surrogate_prediction[i - 1]], axis=0)
                training_y3 = np.append(training_y3, [true_next[2]])
                gpr_3.fit(training_x3, training_y3)

        surrogate_prediction[i] = [x, y, z]

    fig, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1)

    surrogate_prediction_y = surrogate_prediction.transpose()

    ax1.plot(truth.t, truth.y[0])
    ax1.plot(truth.t, surrogate_prediction_y[0])

    ax2.plot(truth.t, truth.y[1])
    ax2.plot(truth.t, surrogate_prediction_y[1])

    ax3.plot(truth.t, truth.y[2])
    ax3.plot(truth.t, surrogate_prediction_y[2])

    plt.show()

    fig2, ((rev_ax1), (rev_ax2), (rev_ax3)) = plt.subplots(3, 1)

    rev_ax1.plot(truth.t, reversion_density[0])

    rev_ax2.plot(truth.t, reversion_density[1])

    rev_ax3.plot(truth.t, reversion_density[2])

    plt.show()

    fig3, ((ax_xx, ax_xy, ax_xz), (ax_yx, ax_yy, ax_yz), (ax_zx, ax_zy, ax_zz)) = plt.subplots(3, 3)

    true_model_points1 = true_model_points1.transpose()
    true_model_points2 = true_model_points2.transpose()
    true_model_points3 = true_model_points3.transpose()

    ax_xx.plot(surrogate_prediction_y[0], surrogate_prediction_y[0], label="truth")
    ax_xx.scatter(true_model_points1[0], true_model_points1[0], label="x model point")
    ax_xx.plot(true_model_points2[0], true_model_points2[0], label="y model point")
    ax_xx.plot(true_model_points3[0], true_model_points3[0], label="z model point")
    ax_xx.set_xlabel("x")
    ax_xx.set_ylabel("x")
    ax_xx.legend()

    ax_xy.plot(surrogate_prediction_y[0], surrogate_prediction_y[1], label="truth")
    ax_xy.plot(true_model_points1[0], true_model_points1[1], label="x model point")
    ax_xy.plot(true_model_points2[0], true_model_points2[1], label="y model point")
    ax_xy.plot(true_model_points3[0], true_model_points3[1], label="z model point")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")

    ax_xz.plot(surrogate_prediction_y[0], surrogate_prediction_y[2], label="truth")
    ax_xz.plot(true_model_points1[0], true_model_points1[2], label="x model point")
    ax_xz.plot(true_model_points2[0], true_model_points2[2], label="y model point")
    ax_xz.plot(true_model_points3[0], true_model_points3[2], label="z model point")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    ax_yx.plot(surrogate_prediction_y[1], surrogate_prediction_y[0], label="truth")
    ax_yx.plot(true_model_points1[1], true_model_points1[0], label="x model point")
    ax_yx.plot(true_model_points2[1], true_model_points2[0], label="y model point")
    ax_yx.plot(true_model_points3[1], true_model_points3[0], label="z model point")
    ax_yx.set_xlabel("y")
    ax_yx.set_ylabel("x")

    ax_yy.plot(surrogate_prediction_y[1], surrogate_prediction_y[1], label="truth")
    ax_yy.plot(true_model_points1[1], true_model_points1[1], label="x model point")
    ax_yy.plot(true_model_points2[1], true_model_points2[1], label="y model point")
    ax_yy.plot(true_model_points3[1], true_model_points3[1], label="z model point")
    ax_yy.set_xlabel("y")
    ax_yy.set_ylabel("y")

    ax_yz.plot(surrogate_prediction_y[1], surrogate_prediction_y[2], label="truth")
    ax_yz.plot(true_model_points1[1], true_model_points1[2], label="x model point")
    ax_yz.plot(true_model_points2[1], true_model_points2[2], label="y model point")
    ax_yz.plot(true_model_points3[1], true_model_points3[2], label="z model point")
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")

    ax_zx.plot(surrogate_prediction_y[2], surrogate_prediction_y[0], label="truth")
    ax_zx.plot(true_model_points1[2], true_model_points1[0], label="x model point")
    ax_zx.plot(true_model_points2[2], true_model_points2[0], label="y model point")
    ax_zx.plot(true_model_points3[2], true_model_points3[0], label="z model point")
    ax_zx.set_xlabel("z")
    ax_zx.set_ylabel("x")

    ax_zy.plot(surrogate_prediction_y[2], surrogate_prediction_y[1], label="truth")
    ax_zy.plot(true_model_points1[2], true_model_points1[1], label="x model point")
    ax_zy.plot(true_model_points2[2], true_model_points2[1], label="y model point")
    ax_zy.plot(true_model_points3[2], true_model_points3[1], label="z model point")
    ax_zy.set_xlabel("z")
    ax_zy.set_ylabel("y")

    ax_zz.plot(surrogate_prediction_y[2], surrogate_prediction_y[2], label="truth")
    ax_zz.plot(true_model_points1[2], true_model_points1[2], label="x model point")
    ax_zz.plot(true_model_points2[2], true_model_points2[2], label="y model point")
    ax_zz.plot(true_model_points3[2], true_model_points3[2], label="z model point")
    ax_zz.set_xlabel("z")
    ax_zz.set_ylabel("z")

    plt.show()