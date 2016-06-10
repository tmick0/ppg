"""
This module contains wrappers for the core PPG filters. All filters support the
same interface -- they have two functions, encode and decode, each of which
accept an image (as a 3d Numpy array) and return an altered image. These filters
make no guarantees that the input images will not be altered as well; if you
need to retain the original, make a copy before invoking any filters.
"""

import numpy as np
from itertools import product
from random import random, randint, randrange
from ppgfilter import *

__all__ = [
    "SubFilter", "UpFilter", "PaethFilter", "BrokenPaethFilter",
    "AverageFilter", "BrokenAverageFilter", "UniformGlitchFilter",
    "RandomLineFilter", "FilterChain", "FilterStack"
]

class AbstractFilter (object):
    """ AbstractFilter -- base class for all Filter objects
    """
    def encode(self, data):
        raise NotImplementedError()
    def decode(self, data):
        raise NotImplementedError()
        
class NativeFilter (AbstractFilter):
    """ NativeFilter -- base class for Filter objects implemented in the C
        module
    """
    def __init__(self):
        raise NotImplementedError()
    def encode(self, data):
        self.f(data, False)
        return data
    def decode(self, data):
        self.f(data, True)
        return data

class SubFilter (NativeFilter):
    """ SubFilter -- The Sub filter from the PNG specification (see
        http://www.libpng.org/pub/png/spec/1.2/PNG-Filters.html). This filter
        will cause glitches to propagate across rows of the image.
    """
    def __init__(self):
        self.f = SubImageFilter

class UpFilter (NativeFilter):
    """ UpFilter -- This is the Up filter from the PNG specification (see
        http://www.libpng.org/pub/png/spec/1.2/PNG-Filters.html). This filter
        will cause glitches to propagate down the columns of an image.
    """
    def __init__(self):
        self.f = UpImageFilter

class PaethFilter (NativeFilter):
    """ PaethFilter -- The Paeth filter from the PNG specification
        (http://www.libpng.org/pub/png/spec/1.2/PNG-Filters.html). Causes errors
        to propagate both to the right and down through an image, resulting in
        a rectangular glitch area. Multiple glitches will quickly accumulate and
        result in harsh noise.
    """
    def __init__(self):
        self.f = PaethImageFilter

class BrokenPaethFilter (NativeFilter):
    """ BrokenPaethFilter -- An intentionally faulty implementation of the Paeth
        Filter, which cause some more interesting results. Refer to
        http://lo.calho.st/projects/adventures-in-image-glitching/ for an
        example.
    """
    def __init__(self):
        self.f = BrokenPaethImageFilter

class AverageFilter (NativeFilter):
    """ BrokenPaethFilter -- The Average filter from the PNG specification
        (http://www.libpng.org/pub/png/spec/1.2/PNG-Filters.html). Causes errors
        to propagate down a diffuse 45-degree diagonal.
    """
    def __init__(self):
        self.f = AverageImageFilter

class BrokenAverageFilter (NativeFilter):
    """ BrokenAverageFilter -- An intentionally faulty Average filter, which
        yields more interesting results. Refer to
        http://lo.calho.st/projects/adventures-in-image-glitching/ for an
        example.
    """
        
    def __init__(self):
        self.f = BrokenAverageImageFilter

class UniformGlitchFilter (NativeFilter):
    """ UniformGlitchFilter -- Introduces disturbances to image data.
    
        Parameters:
        
            - rate (optional): The probability that any particular pixel will be
              altered. Default: 0.005.
             
            - seed (optional): Sets a seed for the filter's internal linear
              congruential generator. Default is randomized.
    """
           
    def __init__(self, rate=0.005, seed=None):
        if seed == None:
            self.seed = randrange(2**32)
        else:
            self.seed = seed
        self.rate = rate
        
    def encode(self, data):
        UniformGlitchImageFilter(data, self.seed, int(self.rate * 2**32))
        return data
        
    def decode(self, data):
        return data

class RandomLineFilter (AbstractFilter):
    """ RandomLineFilter -- Randomly applies different candidate filters to each
        row of the image.
        
        Parameters:
        
            - correlation (optional): Sets the correlation between two rows of
              the image, i.e. the probability that the filter won't change
              between two adjacent rows. Default: 0.5.
              
            - candidates (optional): A list of of filters to be chosen from,
              defined by symbolic names. A filter may be listed multiple times,
              in which case the rate it is chosen will be proportional to its
              number of appearances. Default: ["Sub", "Up", "Paeth", "Average",
              "BrokenPaeth", "BrokenAverage"].
          
            - seed (optional): A seed for the filter's internal linear
              congruential generator. Default is randomized.
    """

    def __init__(self, correlation=0.5, candidates=None, seed=None):
        if seed is None:
            self.seed = randrange(2**32)
        else:
            self.seed = seed
        self.corr = correlation
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
    """ FilterChain -- Allows multiple filters to be applied consecutively. Each
        filter will have its encode stage run, then its decode stage immediately
        after. The filters will be run in the order they are provided.
        
        Parameters:
        
            - Accepts several arguments, each one being an instantiated filter
              to add to the chain.
    """
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
    """ FilterStack -- Allows multiple filters to be applied "together." First, 
        each filter will have its encode stage executed, in the order the
        filters are provided to the stack. Then, each filter's decode stage will
        run, but in the reverse order.
        
        Parameters:
        
            - Accepts several arguments, each one being an instantiated filter
              to add to the stack.
    """
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
