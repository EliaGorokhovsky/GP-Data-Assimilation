import numpy as np
import matplotlib.pyplot as plt
from FixedGPR import RBF, MultiGPR
from scipy.stats import linregress
from odelibrary import L63

initial = np.array([1, 1, 1])
system = L63()
surrogate_run_threshold = 5000
cov_threshold = 1e-3

def forward_map(x, dt=0.01):
    global system
    return x + dt * system.rhs(x, 0)

def cov_norm(cov):
    return np.max(cov)

if __name__ == "__main__":

    num_lengthscales = 10
    lengthscales = np.empty(num_lengthscales)
    numbers = np.empty(num_lengthscales)
    lengthscales[0] = 32
    for i in range(1, num_lengthscales):
        lengthscales[i] = lengthscales[i - 1] / 2

    # Run online learning
    for i in range(len(lengthscales)):
        print("starting", i)
        # Define GPR
        kernel = RBF(1.0, lengthscales[i])
        gpr = MultiGPR(3, kernel, alpha=1e-10)

        measure_pos = initial
        number_trues = 0
        surrogate_run = 0
        while surrogate_run < surrogate_run_threshold:
            # Try using gpr
            next_candidate = gpr.predict([measure_pos])
            next_mean = np.array([coord.mean[0] for coord in next_candidate]).flatten()
            next_cov = cov_norm([coord.cov[0][0] for coord in next_candidate])
            if next_cov > cov_threshold:
                surrogate_run = 0
                true_next = forward_map(measure_pos)
                #print("Used true model at", measure_pos, " candidate", next_mean, " total", number_trues, " cov",
                #      next_cov)
                gpr.add_fit(measure_pos, true_next)
                measure_pos = true_next
                number_trues += 1
            else:
                #print("Used surrogate at", measure_pos, " candidate", next_mean, " cov", next_cov)
                surrogate_run += 1
                measure_pos = next_mean
        print("run", i, number_trues)
        numbers[i] = number_trues

    x_values = np.log(lengthscales)
    y_values = np.log(numbers)
    slope, intercept, rvalue, pvalue, stderr = linregress(x_values, y_values)
    print("regression y = ", slope, "x +", intercept, " with r^2 ", rvalue ** 2)
    #
    ax = plt.axes()
    ax.scatter(x_values, y_values, label="data")
    ax.plot(x_values, slope * x_values + intercept, color="red", label="regression line")
    ax.legend()
    ax.set_title(r'$n$ points vs $\ell$')
    ax.set_xlabel(r'$\log(\ell)$')
    ax.set_ylabel(r'$\log(n)$')
    plt.show()