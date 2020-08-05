from FixedGPR import RBF, FixedGPR
import sklearn.gaussian_process as gauss
import numpy as np
import matplotlib.pyplot as plt

def function(x):
   return np.sin(np.sin(x))

def fun(X):
    return np.array([function(x) for x in X])

if __name__ == "__main__":
    # Generate data
    number = 100

    data = np.array([10 * np.random.random() + 0.1 for i in range(number)])

    test_data = np.arange(0.1, 10, 0.02)
    test_y = fun(test_data)

    test_kernel = gauss.kernels.RBF(1.0 / number)
    test_gpr = gauss.GaussianProcessRegressor(test_kernel, optimizer=None)
    test_gpr.fit(data.reshape(-1, 1), fun(data))
    sklearn_test_outputs = test_gpr.predict(test_data.reshape(-1, 1))

    kernel = RBF(1.0, 1.0 / number)
    gpr = FixedGPR(kernel, alpha=1e-10)
    gpr.fit(data.reshape(-1, 1), fun(data))
    test_outputs = gpr.predict(test_data.reshape(-1, 1)).mean

    incrementalGPR = FixedGPR(kernel, alpha=1e-10)
    for x in data:
        incrementalGPR.add_fit([x], function(x))
    incremental_test_outputs = incrementalGPR.predict(test_data.reshape(-1, 1)).mean

    ax = plt.gca()
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Test of GPR implementations")
    ax.scatter(data, fun(data))
    # ax.plot(test_data, incremental_test_outputs, label="incremental")
    ax.plot(test_data, test_outputs, label="rw2006")
    ax.plot(test_data, sklearn_test_outputs, label="sklearn")
    ax.legend()
    plt.show()
