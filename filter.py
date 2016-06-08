import numpy as np
from itertools import product
from random import random, randint

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

class NullFilter (KernelFilter):
    def kernel(self, data, decode=False):
        return data

class PaethFilter (KernelFilter):

    def kernel(self, data, decode=False):
        if decode:
            copy = data
        else:
            copy = np.copy(data)
        height, width, channels = data.shape
        for c, j, i in product(xrange(channels), xrange(1, height), xrange(1, width)):
            A = data[j,   i-1, c]
            B = data[j-1, i,   c]
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

class SubFilter (KernelFilter):

    def kernel(self, data, decode=False):
        if decode:
            copy = data
        else:
            copy = np.copy(data)
        height, width, channels = data.shape
        for c, j, i in product(xrange(channels), xrange(1, height), xrange(1, width)):
            A = data[j, i-1, c]
            if decode:
                copy[j, i, c] += A
            else:
                copy[j, i, c] -= A
        return copy

class UniformGlitchFilter (NullFilter):
    
    def __init__(self, rate=0.005):
        self.rate = rate
    
    def encode(self, data):
        height, width, channels = data.shape
        for c, j, i in product(xrange(channels), xrange(height), xrange(width)):
            if random() < self.rate:
                data[j, i, c] = randint(0, 255)
        return data
    
