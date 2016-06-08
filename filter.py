import numpy as np
from random import random, randint

class AbstractFilter (object):
    def encode(self, data):
        raise NotImplementedError()
    def decode(self, data):
        raise NotImplementedError()

class PaethFilter (AbstractFilter):

    def kernel(self, data, decode=False):
        if decode:
            copy = data
        else:
            copy = np.copy(data)
        width, height, channels = data.shape
        for c in xrange(channels):
            for i in xrange(1, height):
                for j in xrange(1, width):
                    A = data[j-1, i,   c]
                    B = data[j,   i-1, c]
                    C = data[j-1, i-1, c]
                    p = A + B - C
                    if abs(A - p) <= abs(B - p) and abs(A - p) <= abs(C- p):
                        x = A
                    elif abs(B - p) <= abs(C - p):
                        x = B
                    else:
                        x = C
                    if decode:
                        copy[j, i, c] += x
                    else:
                        copy[j, i, c] -= x
        return copy

    def encode(self, data):
        return self.kernel(data)
    
    def decode(self, data):
        return self.kernel(data, decode=True)

class UniformGlitchFilter (AbstractFilter):
    
    def __init__(self, rate=0.005):
        self.rate = rate
    
    def encode(self, data):
        width, height, channels = data.shape
        for c in xrange(channels):
            for i in xrange(1, height):
                for j in xrange(1, width):
                    if random() < self.rate:
                        data[j, i, c] = randint(0, 255)
        return data
    
    def decode(self, data):
        return data
