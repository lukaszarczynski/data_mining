import numpy as np
import math

a = np.arange(1, 101)
b = np.arange(1, 101, 2)
_c0, _c1 = np.arange(-np.pi, 0, 0.01*np.pi), np.arange(0.01*np.pi, np.pi+0.01, 0.01*np.pi)
c = np.concatenate([_c0, [0], _c1])
d = np.concatenate([_c0, _c1])
e = np.array([abs(np.sin(x) * (np.sin(x)>0)) for x in a])

print(a, b, c, d, e, sep="\n")

A = np.reshape(a, (10, 10))
B = np.diag(a) + np.diag(np.arange(99, 0, -1), k=1) + np.diag(np.arange(99, 0, -1), k=-1)
C = np.triu(np.ones((10, 10)))
D = np.array([[int(x*(x+1)/2) for x in a], [math.factorial(x) for x in a]])
E_ = np.tile(a, (100, 1))
E = np.array([x == 0 for x in np.remainder(E_, E_.T)]).astype(int)

print(A, B, C, D, E, sep="\n")