import numpy as np
from itertools import product
from random import random, randint, randrange
from ppgfilter import SubImageFilter, UpImageFilter, PaethImageFilter, AverageImageFilter, RandomLineImageFilter, UniformGlitchImageFilter

class AbstractFilter (object):
    def encode(self, data):
        raise NotImplementedError()
    def decode(self, data):
        raise NotImplementedError()
        
class NativeFilter (AbstractFilter):
    def __init__(self):
        raise NotImplementedError()
    def encode(self, data):
        self.f(data, False)
        return data
    def decode(self, data):
        self.f(data, True)
        return data

class SubFilter (NativeFilter):
    def __init__(self):
        self.f = SubImageFilter

class UpFilter (NativeFilter):
    def __init__(self):
        self.f = UpImageFilter

class PaethFilter (NativeFilter):
    def __init__(self):
        self.f = PaethImageFilter

class AverageFilter (NativeFilter):
    def __init__(self):
        self.f = AverageImageFilter

class UniformGlitchFilter (NativeFilter):
    def __init__(self, rate=0.005):
        self.seed = randrange(2**32)
        self.rate = rate
    def encode(self, data):
        UniformGlitchImageFilter(data, self.seed, int(self.rate * 2**32))
        return data
    def decode(self, data):
        return data

class RandomLineFilter (AbstractFilter):
    def __init__(self, corr=0.5, candidates=None):
        self.seed = randrange(2**32)
        self.corr = corr
        if candidates == None:
            candidates = ["Sub", "Up", "Paeth", "Average", "BrokenPaeth", "BrokenAverage"]
        self.candidates = candidates
    def encode(self, data):
        RandomLineImageFilter(data, self.candidates, self.seed, int(self.corr * 2**32), False)
        return data
    def decode(self, data):
        RandomLineImageFilter(data, self.candidates, self.seed, int(self.corr * 2**32), True)
        return data

class FilterChain (AbstractFilter):
    def __init__(self, *filters):
        self.filters = filters
    def encode(self, data):
        for f in self.filters:
            data = f.encode(data)
            data = f.decode(data)
        return data
    def decode(self, data):
        return data

class FilterStack (AbstractFilter):
    def __init__(self, *filters):
        self.filters = filters
    def encode(self, data):
        for f in self.filters:
            data = f.encode(data)
        for f in reversed(self.filters):
            data = f.decode(data)
        return data
    def decode(self, data):
        return data
