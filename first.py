from matplotlib import pyplot as plt
import numpy as np
import time

def PerceptronLearning(bemenet, helyes_osztalyozas):
    N, n = bemenet.shape
    lr = 0.1
    w = np.random.randn(n, 1)
    print(w)
    E = 1

    plt.ion()
    figure = plt.figure()
    figure.suptitle('Elso feladat')
    plt.xlim((-1, 1.5))
    plt.ylim((-1, 1.5))
    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(bemenet[:, 1], bemenet[:, 2])

    x = np.linspace(-5, 5, 50)

    while E != 0:
        E = 0

        for i in range(N):
            if np.dot(bemenet[i], w) < 0:
                yi = 0
            else:
                yi = 1

            ei = helyes_osztalyozas[i] - yi
            w += lr * ei * bemenet[i].reshape(n, 1)
            E += ei ** 2

        a = [0, -w[0] / w[2]]
        c = [-w[0] / w[1], 0]
        m = (a[1] - a[0]) / (c[1] - c[0])

        egyenes, = plt.plot(x, x * m + a[1], 'r')
        egyenes.set_ydata(x * m + a[1])
        figure.canvas.draw()
        time.sleep(.25)
        egyenes.remove()
        figure.canvas.flush_events()


bemenet = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

helyes_osztalyozas = [0, 0, 0, 1]

PerceptronLearning(bemenet, helyes_osztalyozas)