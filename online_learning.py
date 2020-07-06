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
surrogate_learning_threshold = 0.001
start_time = 0
end_time = 500
plot_length = 20
spin_up = 20

if __name__ == "__main__":
    # pre_gpr = gauss.GaussianProcessRegressor(
    #     kernel=ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e6)) + WhiteKernel(1.0, (1e-10, 1e6)),
    #     n_restarts_optimizer=10)

    # Gather initial training data
    training_timeseries = integrate.solve_ivp(lambda t, S: system.rhs(S, t), (training_start, training_end), training_initial, Forward_Euler)
    training_indices = np.random.randint(0, len(training_timeseries) - 1, training_quantity)
    data = np.transpose(training_timeseries.y)
    training_x = np.take(data, training_indices, axis=0)
    training_y = np.take(data, training_indices + 1, axis=0)

    # pre_gpr.fit(training_x, training_y)
    #
    # print("Kernel ", pre_gpr.kernel_)
    surrogate_gpr = gauss.GaussianProcessRegressor(kernel=10**2 * RBF(length_scale=25), optimizer=None)

    # Gather truth
    truth = integrate.solve_ivp(lambda t, S: system.rhs(S, t), (start_time, end_time), initial, Forward_Euler)

    # Plot results
    current_time = start_time
    current_state = initial
    training_x = np.array([initial])
    training_y = np.array([initial + dt * system.rhs(current_state, 0)])

    plt.show()
    plt.ion()

    axes = plt.gca()
    while current_time <= end_time - plot_length:
        axes.set_xlim(current_time, current_time + plot_length)
        true_line, = axes.plot(truth.t, truth.y[0], color="black")
        predicted_times = np.array([])
        predicted_states = np.array([])
        simulated_times = np.array([])
        simulated_states = np.array([])
        predicted_points = axes.scatter(predicted_times, predicted_states, color="blue")
        simulated_points = axes.scatter(simulated_times, simulated_states, color="orange")
        plt.draw()
        # On-line surrogate learning
        learning_time = current_time
        for i in range(int(plot_length / dt)):
            # Try to predict the next state using GP
            (next_state, std) = surrogate_gpr.predict([current_state], return_std=True)
            next_state = next_state[0]
            if std > surrogate_learning_threshold:
                true_next_state = current_state + dt * system.rhs(current_state, learning_time)
                training_x = np.append(training_x, [current_state], axis=0)
                training_y = np.append(training_y, [true_next_state], axis=0)
                if learning_time > spin_up:
                    surrogate_gpr.fit(training_x, training_y)
                print(" std ", std)
                simulated_times = np.append(simulated_times, learning_time)
                simulated_states = np.append(simulated_states, true_next_state[0])
                current_state = true_next_state
            else:
                predicted_times = np.append(predicted_times, learning_time)
                predicted_states = np.append(predicted_states, next_state[0])
                current_state = next_state
            learning_time += dt
            predicted_points.set_offsets(np.c_[predicted_times, predicted_states])
            simulated_points.set_offsets(np.c_[simulated_times, simulated_states])
            plt.draw()
            plt.pause(1e-17)
            time.sleep(0.01)


        plt.waitforbuttonpress()
        current_time += plot_length



