import skimage, skimage.io

class AbstractCodec (object):
    def encode(self, data, filename):
        raise NotImplementedError()
    def decode(self, filename):
        raise NotImplementedError()

class SciKitCodec (AbstractCodec):
    def encode(self, data, filename):
        return skimage.io.imsave(filename, data)
    def decode(self, filename):
        return skimage.img_as_ubyte(skimage.io.imread(filename))
