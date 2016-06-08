import sys
import numpy as np
from codec import SciKitCodec
from filter import PaethFilter, SubFilter, UniformGlitchFilter

np.seterr(all='ignore')

def main(infile, outfile):
    codec  = SciKitCodec()
    paeth  = PaethFilter()
    sub    = SubFilter()
    glitch = UniformGlitchFilter(rate=0.00002)
    
    f = codec.decode(infile)
    
    f = sub.encode(f)
    f = glitch.encode(f)
    f = glitch.encode(f)
    f = sub.decode(f)
    
    f = paeth.encode(f)
    f = glitch.encode(f)
    f = paeth.decode(f)
    
    codec.encode(f, outfile)
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
