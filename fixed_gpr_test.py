from FixedGPR import RBF, FixedGPR
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Generate data
    number = 10
    slope = np.random.random()
    data = np.array([np.random.random() for i in range(number)])
    print(data)

    kernel = RBF(1.0, 0.01 / number)
    gpr = FixedGPR(kernel)
    gpr.fit(data.reshape(-1, 1), slope * data)

    test_data = np.arange(0, 1, 0.1)
    test_outputs = gpr.predict(test_data.reshape(-1, 1)).mean

    incrementalGPR = FixedGPR(kernel)
    for x in data:
        incrementalGPR.add_fit([x], slope * x)
    incremental_test_outputs = incrementalGPR.predict(test_data.reshape(-1, 1)).mean

    ax = plt.gca()
    ax.scatter(data, slope * data)
    ax.plot(test_data, incremental_test_outputs)
    plt.show()
