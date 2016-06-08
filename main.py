import sys
import numpy as np
from codec import SciKitCodec
from filter import PaethFilter, UniformGlitchFilter

np.seterr(all='ignore')

def main(infile, outfile):
    codec = SciKitCodec()
    paeth = PaethFilter()
    glitch = UniformGlitchFilter(rate=0.00005)
    
    f = codec.decode(infile)
    f = paeth.encode(f)
    f = glitch.encode(f)
    f = paeth.decode(f)
    
    codec.encode(f, outfile)
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
