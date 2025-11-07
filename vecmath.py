import math

TOL = 1e-9

def dot(a, b):
    dot = 0
    for i in range(len(a)):
        dot += a[i] * b[i]
    return dot
def sub(a,b):
    sublist = []
    for i in range(len(a)):
        sublist.append(a[i] - b[i])
    return tuple(sublist)
def add(a,b):
    add = []
    for i in range(len(a)):
        add.append(a[i] + b[i])
    return tuple(add)
def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)

def length(a): return math.sqrt(dot(a,a))
def norm(a):
    L = length(a)
    if L < TOL: return (0.0, 0.0, 0.0)
    norm = []
    for i in range(len(a)):
        norm.append(a[i]/L)
    return tuple(norm)

class Ray:
    def __init__(self, origin, direction):
        d = norm(direction)
        self.o = (float(origin[0], float(origin[1]), float(origin[2])))
        self.d = d
    def fpt(self, T: float) -> float:
        (ox, oy, oz) = self.o
        (dx, dy, dz) = self.d
        return float(self.s.evaluate(ox + T*dx, oy + T*dy, oz + T*dz))
