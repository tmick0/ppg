import sys
from ppg.codec import SciKitCodec
from ppg.filter import *
from ppg.parser import LoadFilter

def main(infile=None, outfile=None, filterdef=None, quiet=""):
    
    # Check command-line arguments
    if infile is None or outfile is None:
        print("usage: %s <infile> <outfile> [filterdef] [quiet]" % (sys.argv[0]))
        return 1
    
    # Default filterdef if None
    if filterdef is None:
        filterdef = "testfilter.txt"
    
    # Set quiet flag
    quiet = (quiet in ["quiet", "q"])

    # Instantiate a wrapper around skimage to make encoding/decoding files easy
    codec = SciKitCodec()

    # Load the file
    f = codec.decode(infile)

    # Load the filter chain
    fh = open(filterdef, "r")
    chain = LoadFilter(fh.read())
    fh.close()
    
    # Apply the filter chain, then save the glitched image
    f = chain.encode(f, not quiet)
    codec.encode(f, outfile)
    
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
