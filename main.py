import sys
import numpy as np
from codec import SciKitCodec
from filter import SubFilter, UniformGlitchFilter, FilterChain, FilterStack

np.seterr(all='ignore')

def main(infile, outfile):
    codec = SciKitCodec()

    chain = FilterChain(
    
        FilterStack(
            SubFilter(),
            UniformGlitchFilter(rate=0.0001)
        ),
    
        #FilterStack(
        #    RandomLineFilter(blacklist=[SubFilter]),
        #    UniformGlitchFilter(rate=0.0005)
        #),
        
        #FilterStack(
        #    SubFilter(),
        #    UniformGlitchFilter(rate=0.00001)
        #)
    
    )
    
    f = codec.decode(infile)
    f = chain.encode(f)
    codec.encode(f, outfile)
    
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
