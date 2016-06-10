import sys
import numpy as np
from codec import SciKitCodec
from filter import SubFilter, UpFilter, PaethFilter, AverageFilter, UniformGlitchFilter, RandomLineFilter, FilterChain, FilterStack

np.seterr(all='ignore')

def main(infile, outfile):
    codec = SciKitCodec()

    chain = FilterChain(
    
#        FilterStack(
#            PaethFilter(),
#            UniformGlitchFilter(rate=0.00001)
#        ),
#    
#        FilterStack(
#            AverageFilter(),
#            UniformGlitchFilter(rate=0.0005)
#        ),
#        
#        FilterStack(
#            UpFilter(),
#            UniformGlitchFilter(rate=0.00001)
#        ),
#        
#        FilterStack(
#            SubFilter(),
#            UniformGlitchFilter(rate=0.00001)
#        ),
    
        FilterStack(
            RandomLineFilter(),
            UniformGlitchFilter(rate=0.0005)
        )
    
    )
    
    f = codec.decode(infile)
    f = chain.encode(f)
    codec.encode(f, outfile)
    
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
