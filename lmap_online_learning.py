import numpy as np
import matplotlib.pyplot as plt
from FixedGPR import RBF, FixedGPR, Normal
import time


kernel = RBF(1.0, 0.05)
r = 4
initial = 0.3
spin_up = 20
online_steps = 1000
cov_threshold = 1e-9


def lmap(r, x):
    """
    Simulates the logistic map with given r (parameter) on xn = x (0 <= x <= 1)
    Chaos achieved at r > 3.56995
    """
    return r * x * (1 - x)

if __name__ == "__main__":
    # Get invariant measure of lmap
    bins = 100
    measure_steps = 10000
    x_values = np.linspace(0, 1, bins)
    values = np.zeros(bins)
    measure_pos = initial
    for i in range(spin_up):
        measure_pos = lmap(r, measure_pos)
        print(measure_pos)
    for i in range(measure_steps):
        values[int(np.floor(measure_pos * bins))] += 1
        measure_pos = lmap(r, measure_pos)
    values /= measure_steps

    fig, ((invariant_ax), (timeseries_ax), (map_ax)) = plt.subplots(3, 1)
    invariant_ax.set_title(r'Invariant measure $\mu$')
    timeseries_ax.set_title("Timeseries")
    map_ax.set_title(r'Recurrence relation $x_{n+1} = rx_n(1 - x_n)$')
    invariant_ax.plot(x_values, values, label="lmap")

    plt.show()
    plt.ion()

    # Define GPR
    gpr = FixedGPR(kernel, alpha=1e-10)

    # Run online learning

    # Dynamic plot timeseries
    plt.draw()

    # Get invariant measure of surrogate
    surrogate_values = np.zeros(bins)
    measure_quantity = 0
    measure, = invariant_ax.plot(x_values, surrogate_values, label="surrogate")
    invariant_ax.legend()

    # Get actual plot of map
    map_x_values = np.linspace(0, 1, 100)
    map_y_values = np.array([lmap(r, x) for x in map_x_values])
    real_lmap, = map_ax.plot(map_x_values, map_y_values, label="lmap")
    predicted_map: Normal = gpr.predict(map_x_values)
    mean = predicted_map.mean
    std = 1 / -np.log(np.diag(predicted_map.cov))
    map_ax.fill_between(map_x_values, mean - std, mean + std, facecolor="grey", alpha=0.5)
    surrogate_lmap, = map_ax.plot(map_x_values, predicted_map.mean, label="surrogate")
    map_ax.set_ylim((-1, 2))
    map_ax.legend()

    plt.waitforbuttonpress()
    number_trues = 0
    for i in range(online_steps):
        # Try using gpr
        next_candidate = gpr.predict([measure_pos])
        surrogate_values[int(np.floor(measure_pos * bins))] += 1
        measure_quantity += 1
        delay = 0
        if next_candidate.cov > cov_threshold or next_candidate.mean[0] >= 1 or next_candidate.mean[0] <= 0:
            true_next = lmap(r, measure_pos)
            gpr.add_fit([measure_pos], true_next)
            measure_pos = true_next
            timeseries_ax.scatter(i, measure_pos, color="yellow")
            number_trues += 1
            print("Used true model at", measure_pos, " candidate", next_candidate, " total", number_trues)
            # Get predicted map
            predicted_map = gpr.predict(map_x_values)
            mean = predicted_map.mean
            std = 1 / -np.log(np.diag(predicted_map.cov))
            surrogate_lmap.set_ydata(mean)
            map_ax.collections.clear()
            map_ax.fill_between(map_x_values, mean - std, mean + std, facecolor="grey", alpha=0.5)
            delay = 0.01
        else:
            measure_pos = next_candidate.mean[0]
            timeseries_ax.scatter(i, measure_pos, color="blue")
            # print("Used surrogate at", measure_pos, " candidate", next_candidate)
            delay = 0.01
        measure.set_ydata(surrogate_values / measure_quantity)

        plt.draw()
        plt.pause(1e-17)
        time.sleep(delay)

