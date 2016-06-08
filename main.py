import sys
import numpy as np
from codec import SciKitCodec
from filter import RandomLineFilter, UniformGlitchFilter

np.seterr(all='ignore')

def main(infile, outfile):
    codec  = SciKitCodec()
    rlf    = RandomLineFilter()
    glitch = UniformGlitchFilter(rate=0.0005)
    
    f = codec.decode(infile)
    
    f = rlf.encode(f)
    f = glitch.encode(f)
    f = rlf.decode(f)
    
    codec.encode(f, outfile)
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
