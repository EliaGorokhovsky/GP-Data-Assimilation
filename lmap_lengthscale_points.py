import numpy as np
import matplotlib.pyplot as plt
from FixedGPR import RBF, FixedGPR
from scipy.stats import linregress

crit_r = 3.5699456718709449018420051513864989367638369115148323781079755299213628875001367775263210342163
r = crit_r
c = 1.401155189
initial = 0.3
online_steps = 1000
cov_threshold = 1e-9


def lmap(r, x):
    """
    Simulates the logistic map with given r (parameter) on xn = x (0 <= x <= 1)
    Chaos achieved at r > 3.5699456718709449018420051513864989367638369115148323781079755299213628875001367775263210342163
    """
    return r * x * (1 - x)

def fmap(c, x):
    """
    Simulates the Feigenbaum map with given c
    """
    return x**2 - c

if __name__ == "__main__":

    num_lengthscales = 60
    lengthscales = np.empty(num_lengthscales)
    numbers = np.empty(num_lengthscales)
    lengthscales[0] = 0.001
    for i in range(1, num_lengthscales):
        lengthscales[i] = lengthscales[i - 1] / 1.025

    # Run online learning
    for i in range(len(lengthscales)):
        print("starting", i)
        # Define GPR
        kernel = RBF(1.0, lengthscales[i])
        gpr = FixedGPR(kernel, alpha=1e-10)
        measure_pos = initial

        number_trues = 0
        for j in range(online_steps):
            # Try using gpr
            next_candidate = gpr.predict([measure_pos])
            if next_candidate.cov > cov_threshold or next_candidate.mean[0] >= 1 or next_candidate.mean[0] <= 0:
                true_next = lmap(r, measure_pos)
                gpr.add_fit([measure_pos], true_next)
                measure_pos = true_next
                number_trues += 1
                # print("Used true model at", measure_pos, " candidate", next_candidate, " total", number_trues)
            else:
                measure_pos = next_candidate.mean[0]
                # print("Used surrogate at", measure_pos, " candidate", next_candidate)
        numbers[i] = number_trues

    x_values = np.log(lengthscales)
    y_values = np.log(numbers)
    slope, intercept, rvalue, pvalue, stderr = linregress(x_values, y_values)
    print("regression y = ", slope, "x +", intercept, " with r^2 ", rvalue ** 2)

    ax = plt.axes()
    ax.scatter(x_values, y_values, label="data")
    ax.plot(x_values, slope * x_values + intercept, color="red", label="regression line")
    ax.legend()
    ax.set_title(r'$n$ points vs $\ell$')
    ax.set_xlabel(r'$\log(\ell)$')
    ax.set_ylabel(r'$\log(n)$')
    plt.show()

