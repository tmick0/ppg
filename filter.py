import numpy as np
from itertools import product
from random import random, randint, WichmannHill

class AbstractFilter (object):
    def encode(self, data):
        raise NotImplementedError()
    def decode(self, data):
        raise NotImplementedError()
        
class KernelFilter (AbstractFilter):
    def kernel(self, data, decode=False):
        raise NotImplementedError()
    def encode(self, data):
        return self.kernel(data)
    def decode(self, data):
        return self.kernel(data, decode=True)

class LineFilter (KernelFilter):
    def kernel(self, data, decode=False):
        if decode:
            dest = data
        else:
            dest = np.copy(data)
        height, width, channels = data.shape
        for c, i in product(xrange(channels), xrange(height)):
            self.line(data, dest, c, i, decode)
        return dest
    def line(self, data, dest, c, i, decode):
        raise NotImplementedError()

class PaethFilter (LineFilter):
    def line(self, data, dest, c, i, decode):
        height, width, channels = data.shape
        if i > 0:
            for j in xrange(1, width):
                A = data[i,   j-1, c]
                B = data[i-1, j,   c]
                C = data[i-1, j-1, c]
                p = A + B - C
                if abs(A - p) <= abs(B - p) and abs(A - p) <= abs(C- p):
                    x = A
                elif abs(B - p) <= abs(C - p):
                    x = B
                else:
                    x = C
                if decode:
                    dest[i, j, c] += x
                else:
                    dest[i, j, c] -= x

class SubFilter (LineFilter):
    def line(self, data, dest, c, i, decode):
        height, width, channels = data.shape
        for j in xrange(1, width):
            A = data[i, j-1, c]
            if decode:
                dest[i, j, c] += A
            else:
                dest[i, j, c] -= A

class UpFilter (LineFilter):
    def line(self, data, dest, c, i, decode):
        height, width, channels = data.shape
        if i > 0:
            for j in xrange(1, width):
                B = data[i-1, j, c]
                if decode:
                    dest[i, j, c] += B
                else:
                    dest[i, j, c] -= B

class NullFilter (LineFilter):
    def line(self, data, dest, c, i, decode):
        pass

class AverageFilter (LineFilter):
    def line(self, data, dest, c, i, decode):
        height, width, channels = data.shape
        if i > 0:
            for j in xrange(1, width):
                A = data[i,   j-1, c]
                B = data[i-1, j,   c]
                x = (A - B) / 2
                if decode:
                    dest[i, j, c] += x
                else:
                    dest[i, j, c] -= x

class RandomLineFilter (LineFilter):
    def __init__(self, seed=None):
        self.rng = WichmannHill(seed)
        self.seed = self.rng.getstate()
    def encode(self, data):
        self.rng.setstate(self.seed)
        return self.kernel(data)
    def decode(self, data):
        self.rng.setstate(self.seed)
        return self.kernel(data, decode=True)
    def line(self, data, dest, c, i, decode):
        candidates = [NullFilter, SubFilter]
        if i > 0:
            candidates.extend([UpFilter, AverageFilter, PaethFilter])
        lf = self.rng.choice(candidates)()
        lf.line(data, dest, c, i, decode)

class UniformGlitchFilter (KernelFilter):
    def __init__(self, rate=0.005):
        self.rate = rate
    def encode(self, data):
        height, width, channels = data.shape
        for c, j, i in product(xrange(channels), xrange(height), xrange(width)):
            if random() < self.rate:
                data[j, i, c] = randint(0, 255)
        return data
    def decode(self, data):
        return data
