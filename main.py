from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from odelibrary import L63
from forward_euler import Forward_Euler
from scipy import integrate
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gauss

perturbation = 0.1
dt = 0.01
training_end_time = 500
testing_end_time = 520


def perturbedL63(S, t, a=10, b=28, c=8 / 3):
    x = S[0]
    y = S[1]
    z = S[2]
    foo_rhs = np.empty(3)
    foo_rhs[0] = -a * x + a * y
    foo_rhs[1] = b * x - y - x * z
    foo_rhs[2] = -c * z + x * y
    return foo_rhs


if __name__ == "__main__":
    # Gather training data
    system = L63()
    timeseries = integrate.solve_ivp(lambda t, S: system.rhs(S, t), (0, training_end_time), np.array([1, 1, 1]), Forward_Euler)
    # print(timeseries)

    # Perturb L63
    perturbed_b = np.random.normal(1.0, perturbation) * 28
    print("Perturbed b:", perturbed_b)

    data = np.transpose(timeseries.y)
    residuals = np.empty((len(data), 3))
    slopes = np.empty((len(data), 3))
    for i in range(len(data) - 1):
        residuals[i] = (data[i + 1] - data[i]) / dt - perturbedL63(data[i], timeseries.t[i], b=perturbed_b)
        slopes[i] = (data[i + 1] - data[i]) / dt

    # residuals_y = residuals[:, 1]
    #
    # GPR
    training_indices = np.random.randint(0, len(data), 1000)
    gpr = gauss.GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e6)) + WhiteKernel(1.0, (1e-10, 1e6)),
        n_restarts_optimizer=5)\
        .fit(np.take(data, training_indices, axis=0), np.take(residuals, training_indices, axis=0))
    print("Model Error Kernel ", gpr.kernel_)

    surrogate_training_indices = np.random.randint(0, len(data), 10000)
    surrogate_gpr = gauss.GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e6)) + WhiteKernel(1.0, (1e-10, 1e6)),
        n_restarts_optimizer=15)\
        .fit(np.take(data, training_indices, axis=0), np.take(slopes, training_indices, axis=0))
    print("Surrogate Kernel ", surrogate_gpr.kernel_)

    def learnedL63(t, S):
        return perturbedL63(S, t, b=perturbed_b) + gpr.predict(S.reshape(1, -1))[0]

    def surrogateL63(t, S):
        return surrogate_gpr.predict(S.reshape(1, -1))[0]


    true_timeseries = integrate.solve_ivp(lambda t, S: system.rhs(S, t), (training_end_time, testing_end_time), data[-1], Forward_Euler)
    guessed_timeseries = integrate.solve_ivp(learnedL63, (training_end_time, testing_end_time), data[-1], Forward_Euler)
    surrogate_timeseries = integrate.solve_ivp(surrogateL63, (training_end_time, testing_end_time), data[-1], Forward_Euler)
    bad_timeseries = integrate.solve_ivp(lambda t, S: perturbedL63(S, t, b=perturbed_b), (training_end_time, testing_end_time), data[-1],
                                         Forward_Euler)

    corrected_error = np.empty(len(true_timeseries.t))
    uncorrected_error = np.empty(len(true_timeseries.t))
    surrogate_error = np.empty(len(true_timeseries.t))
    for i in range(len(true_timeseries.t)):
        corrected_error[i] = np.sqrt((guessed_timeseries.y[0][i] - true_timeseries.y[0][i]) ** 2 + (
                    guessed_timeseries.y[1][i] - true_timeseries.y[1][i]) ** 2 + (
                                                 guessed_timeseries.y[2][i] - true_timeseries.y[2][i]) ** 2)
        uncorrected_error[i] = np.sqrt((bad_timeseries.y[0][i] - true_timeseries.y[0][i]) ** 2 + (
                bad_timeseries.y[1][i] - true_timeseries.y[1][i]) ** 2 + (
                                             bad_timeseries.y[2][i] - true_timeseries.y[2][i]) ** 2)
        surrogate_error[i] = np.sqrt((surrogate_timeseries.y[0][i] - true_timeseries.y[0][i]) ** 2 + (
                surrogate_timeseries.y[1][i] - true_timeseries.y[1][i]) ** 2 + (
                                               surrogate_timeseries.y[2][i] - true_timeseries.y[2][i]) ** 2)

    # print("True: ", true_timeseries)
    # print("Guessed: ", guessed_timeseries)
    # print("Bad: ", bad_timeseries)

    fig, ((ax3), (ax4)) = plt.subplots(2, 1)

    # subsample_x = np.take(data, training_indices, axis=0)
    # subsample_y = gpr.predict(subsample_x)[:, 1]
    # plot_x = subsample_x[:, 0]
    # ax1.scatter(plot_x, np.take(residuals_y, training_indices), label="training data")
    # ax1.scatter(plot_x, subsample_y, label="prediction")
    # ax1.legend()

    # domain = np.arange(-20, 20, 1)
    # xs = np.array([np.zeros(len(domain)), domain, np.zeros(len(domain))]).transpose()
    # ys = gpr.predict(xs)[:, 1]
    # real_ys = [(perturbedL63([i, 0, 0], 0) - perturbedL63([i, 0, 0], 0, b=perturbed_b))[1] for i in domain]
    # ax2.plot(domain, ys, label="prediction")
    # ax2.scatter(domain, real_ys, label="residuals")
    # ax2.legend()

    ax3.plot(true_timeseries.t, true_timeseries.y[1], label="true")
    ax3.plot(guessed_timeseries.t, guessed_timeseries.y[1], label="corrected")
    ax3.plot(bad_timeseries.t, bad_timeseries.y[1], label="prior")
    ax3.plot(surrogate_timeseries.t, surrogate_timeseries.y[1], label="surrogate")
    ax3.set_title("Timeseries y")
    ax3.legend()

    ax4.plot(true_timeseries.t, corrected_error, label="corrected")
    ax4.plot(true_timeseries.t, uncorrected_error, label="prior")
    ax4.plot(true_timeseries.t, surrogate_error, label="surrogate")
    ax4.legend()
    ax4.set_title("Error over time")
    plt.show()

    # training_points = np.take(data, surrogate_training_indices, axis=0).transpose()
    # ax = plt.axes(projection='3d')
    # ax.scatter(training_points[0], training_points[1], training_points[2], alpha=0.05)
    # ax.scatter(true_timeseries.y[0], true_timeseries.y[1], true_timeseries.y[2], cmap="hot", c=true_timeseries.t)
    # plt.show()
