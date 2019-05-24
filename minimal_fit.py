import numpy as np
import matplotlib.pyplot as plt

def descent(x, y, epochs, step, order):
    weights, d, u = np.zeros(order + 1), [], np.linspace(min(x)-1,max(x)+1,100)
    d.extend(x**i for i in np.arange(order+1))
    features = np.column_stack(d)

    for i in range(epochs):
        est, cost = 0, 0
        difference = sum([(weights[i] * (x ** i)) for i in range(order + 1)]).T - y
        weights = weights + (-step * (1/len(y)) * np.matmul(difference, features))
        cost = ((1/(2*(len(y)))) * np.sum(sum([((y - (weights[i] * (x ** i))) ** 2) for i in range(order + 1)]) ** 2))

    plt.scatter(x,y)
    plt.plot(u, sum([(u ** i) * weights[i] for i in range(order + 1)]), 'r-')
    plt.show()

descent(np.random.randint(0,20,(4)), np.random.randint(0,20,(4)), 90000, 0.0000001, 3)
