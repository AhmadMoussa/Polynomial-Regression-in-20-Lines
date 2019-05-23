import numpy as np
import matplotlib.pyplot as plt

def descent(x, y, epochs, step, order):
    weights, d, u = np.zeros(order + 1), [], np.linspace(min(x)-1,max(x)+1,100)
    d.append(np.ones([1,len(x)])[0])
    for i in np.arange(1, order+1):
        d.append(x ** (i))
    features = np.column_stack(d)

    for i in range(epochs):
        est, cost = 0, 0
        est = sum([(weights[i] * (x ** i)) for i in range(order + 1)])
        difference = est.T - y
        weights = weights + (-step * (1/len(y)) * np.matmul(difference, features))
        cost = ((1/(2*(len(y)))) * np.sum(sum([((y - (weights[i] * (x ** i))) ** 2) for i in range(order + 1)]) ** 2))

    plt.scatter(x,y)
    plt.plot(u, sum([(u ** i) * weights[i] for i in range(order + 1)]), 'r-')
    plt.show()

descent(np.random.randint(0,20,(20)), np.random.randint(0,20,(20)), 800000, 0.0000001, 3)
