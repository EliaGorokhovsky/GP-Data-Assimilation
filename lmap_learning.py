from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gauss

num_training_iterations = 20
num_testing_iterations = 50
r = 3.9

def lmap(r, x):
    """
    Simulates the logistic map with given r (parameter) on xn = x (0 <= x <= 1)
    Chaos achieved at r > 3.56995
    """
    return r * x * (1 - x)

if __name__ == "__main__":
    # Generate data
    x_init = 0.5
    data = np.empty(num_training_iterations + 1)
    data[0] = x_init
    for i in range(num_training_iterations):
        data[i + 1] = lmap(r, data[i])

    # Learn model
    # training_indices = np.random.randint(0, len(data), 1000)
    gpr = gauss.GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e6)) + WhiteKernel(1.0, (1e-10, 1e6)),
        n_restarts_optimizer=5) \
        .fit(data[:-1].reshape(-1, 1), data[1:].reshape(-1, 1))
        #.fit(np.take(data, training_indices, axis=0), np.take(residuals, training_indices, axis=0))
    print("Kernel: ", gpr.kernel_)

    def learned_lmap(x):
        return gpr.predict([[x]])[0]

    # Generate training data
    correct_data = np.empty(num_testing_iterations + 1)
    correct_data[0] = data[-1]
    predicted_data = np.empty(num_testing_iterations + 1)
    predicted_data[0] = data[-1]
    errors = np.empty(num_testing_iterations + 1)
    errors[0] = 0
    for i in range(num_testing_iterations):
        correct_data[i + 1] = lmap(r, correct_data[i])
        predicted_data[i + 1] = learned_lmap(predicted_data[i])
        errors[i + 1] = np.abs(correct_data[i + 1] - predicted_data[i + 1])

    # Plot results
    fig, ((ax1), (ax2)) = plt.subplots(2, 1)

    ax1.plot(np.arange(num_training_iterations, len(correct_data) + num_training_iterations), correct_data, label="truth")
    ax1.plot(np.arange(num_training_iterations, len(predicted_data) + num_training_iterations), predicted_data, label="prediction")
    ax1.legend()
    ax1.set_title("timeseries")

    ax2.plot(np.arange(0, len(errors)), errors)
    ax2.set_title("error")

    plt.show()