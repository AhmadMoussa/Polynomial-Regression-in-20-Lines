import numpy as np
import matplotlib.pyplot as plt

def descent(x, y, epochs, step, order):
    weights = np.zeros(order + 1)
    d = []
    d.append(np.ones([1,len(x)])[0])
    for i in np.arange(1, order+1):
        d.append(x ** (i))
    features = np.column_stack(d)

    for i in range(epochs):
        est, cost = 0, 0
        for i in range(order + 1):
            est += (weights[i] * (x ** i))
        difference = est.T - y
        weights = weights + (-step * (1/len(y)) * np.matmul(difference, features))
        for i in range(order + 1):
            cost += ((y - (weights[i] * x ** i)) ** 2)
        cost = ((1/(2*(len(y)))) * np.sum(cost ** 2))

    plt.scatter(x,y)
    u = np.linspace(0,3,100)
    plt.plot(u, sum([(u ** i) * weights[i] for i in range(order + 1)]), 'r-')
    plt.show()

descent(np.asarray([1,2,0,3]), np.asarray([0,1,2,3]), 100000, 0.001, 4)
