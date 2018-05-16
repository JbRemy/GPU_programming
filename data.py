import numpy as np
from numpy.random import normal, uniform
from math import pi, floor


def generate_and_save_data(d, N, P):

    for n in range(N):
        A = normal(0, 1, size=(d, d))
        np.savetxt('files/%i/A.txt' % n, A, delimiter=';')
        for p in range(P):
            theta = uniform(0, 2*pi)
            J = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            np.savetxt('files/%i/J_%i.txt' % (n, p), J, delimiter=';')


def test_results(n, d, N, P):

    pos_i = [0]*P
    pos_j = [0]*P
    shift = 0
    for i in range(P):
        pos_i[i] = floor((i + shift) / d)
        pos_j[i] = (i + shift) % d

    if (i + shift) % d == d-1:
        shift += 1

    score = 0

    for _ in range(n):

        i = np.random.randint(N)
        A = np.load('files/%i/A.txt' % i)
        for p in range(P):
            J_temp = np.load('files/%i/J_%i.txt' % (n, p))
            J = np.identity(d)
            i = pos_i[p]
            j = pos_j[p]
            J[[[i, d-j], [j, d-i]]] = J_temp

            A = np.dot(J, A)

        res = np.load('files/%i/out.txt' % i)

        if res != A:
            score += 1

    score = score / N

    return score

