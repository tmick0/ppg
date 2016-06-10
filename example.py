import sys
from ppg.codec import SciKitCodec
from ppg.filter import *

def main(infile=None, outfile=None, scale=None):
    
    # Check command-line arguments
    if infile is None or outfile is None:
        print("usage: %s <infile> <outfile> [scalar]" % (sys.argv[0]))
        return 1
    
    # Default scalar if None
    if scale is None:
        scale = 1.0
    else:
        scale = float(scale)

    # Instantiate a wrapper around skimage to make encoding/decoding files easy
    codec = SciKitCodec()

    # Load the file
    f = codec.decode(infile)

    # Define a filter chain
    chain = FilterChain(
    
        # First, we will do a randomized set of PNG-style filters with 95% line
        # correlation, altering pixels at a rate of 0.04%.
        FilterStack(
            RandomLineFilter(correlation=0.95, candidates=["Average","Paeth","Null","Up","Sub"]),
            UniformGlitchFilter(rate=0.00040*scale)
        ),
        
        # Next, apply our BrokenPaethFilter and alter pixels at 0.001%.
        FilterStack(
            BrokenPaethFilter(),
            UniformGlitchFilter(rate=0.00001*scale)
        ),

        # Finally, apply the BrokenAverageFilter and alter pixels at 0.025%.
        FilterStack(
            BrokenAverageFilter(),
            UniformGlitchFilter(rate=0.00025*scale)
        )
    
    )
    
    # Apply the filter chain, then save the glitched image
    f = chain.encode(f)
    codec.encode(f, outfile)
    
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
