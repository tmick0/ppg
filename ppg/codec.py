""" Contains convenience methods for encoding and decoding image files
    from/to Numpy arrays.
"""

import skimage, skimage.io

class AbstractCodec (object):
    """ AbstractCodec -- base class for image codecs
    """
    def encode(self, data, filename):
        raise NotImplementedError()
    def decode(self, filename):
        raise NotImplementedError()

class SciKitCodec (AbstractCodec):
    """ SciKitCodec -- uses skimage to load and save files
    """
    def encode(self, data, filename):
        return skimage.io.imsave(filename, data)
    def decode(self, filename):
        return skimage.img_as_ubyte(skimage.io.imread(filename))
