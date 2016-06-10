"""
This module contains wrappers for the core PPG filters. All filters support the
same interface -- they have two functions, encode and decode, each of which
accept an image (as a 3d Numpy array) and return an altered image. These filters
make no guarantees that the input images will not be altered as well; if you
need to retain the original, make a copy before invoking any filters.
"""

import numpy as np
from collections import deque
from datetime import datetime
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
    def encode(self, data, verbose=False):
        self.f(data, False)
        return data
    def decode(self, data, verbose=False):
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
        
    def encode(self, data, verbose=False):
        UniformGlitchImageFilter(data, self.seed, int(self.rate * 2**32))
        return data
        
    def decode(self, data, verbose=False):
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
    def encode(self, data, verbose=False):
        RandomLineImageFilter(data, self.candidates, self.seed, int(self.corr * 2**32), False)
        return data
    def decode(self, data, verbose=False):
        RandomLineImageFilter(data, self.candidates, self.seed, int(self.corr * 2**32), True)
        return data

class TrackingFilter (AbstractFilter):

    def prechain(self, t, v, d):
        if v is True:
            self._start = datetime.now()
            print("%s[+] %s starting" % ("    " * d, type(t).__name__))
        
    def postchain(self, t, v, d):
        if v is True and d == 0:
            delta = datetime.now() - self._start
            print("%s[+] %s finished in %.3fs." % ("    " * d, type(t).__name__, delta.total_seconds()))

    def preencode(self, t, v):
        if v is True:
            self.tracking.append(("+", t, datetime.now()))
            
    def predecode(self, t, v):
        if v is True:
            self.tracking.append(("-", t, datetime.now())) 
            
    def postencode(self, t, v, d):
        if v is True:
            m, tt, k = self.tracking.pop()
            if (m, tt) != ("+", t):
                raise RuntimeError("bad tracking state")
            delta = datetime.now() - k
            print("%s[+] Encoded with %s in %.3fs." % ("    " * (d+1), type(t).__name__, delta.total_seconds()))
    
    def postdecode(self, t, v, d):
        if v is True:
            m, tt, k = self.tracking.pop()
            if (m, tt) != ("-", t):
                raise RuntimeError("bad tracking state")
            delta = datetime.now() - k
            print("%s[-] Decoded with %s in %.3fs." % ("    " * (d+1), type(t).__name__, delta.total_seconds()))

class FilterChain (TrackingFilter):
    """ FilterChain -- Allows multiple filters to be applied consecutively. Each
        filter will have its encode stage run, then its decode stage immediately
        after. The filters will be run in the order they are provided.
        
        Parameters:
        
            - Accepts several arguments, each one being an instantiated filter
              to add to the chain.
    """
    def __init__(self, *filters):
        self.filters = filters
        self.tracking = deque()
    def encode(self, data, verbose=False, depth=0):
        self.prechain(self, verbose, depth)
        for f in self.filters:
            self.preencode(f, verbose)
            data = f.encode(data, verbose, depth+1)
            self.postencode(f, verbose, depth)
            self.predecode(f, verbose)
            data = f.decode(data, verbose, depth+1)
            self.postdecode(f, verbose, depth)
        self.postchain(self, verbose, depth)
        return data
    def decode(self, data):
        return data

class FilterStack (TrackingFilter):
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
        self.tracking = deque()
        
    def encode(self, data, verbose=False, depth=0):
        self.prechain(self, verbose, depth)
        for f in self.filters:
            self.preencode(f, verbose)
            data = f.encode(data)
            self.postencode(f, verbose, depth)
        for f in reversed(self.filters):
            self.predecode(f, verbose)
            data = f.decode(data)
            self.postdecode(f, verbose, depth)
        self.postchain(self, verbose, depth)
        return data
        
    def decode(self, data, verbose=False, depth=0):
        return data
    

