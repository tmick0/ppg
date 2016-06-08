import sys
import numpy as np
from codec import SciKitCodec
from filter import PaethFilter

np.seterr(all='ignore')

def main(infile, outfile):
    codec = SciKitCodec()
    paeth = PaethFilter()
    
    raw = codec.decode(infile)
    paeth_encoded = paeth.encode(raw)
    paeth_decoded = paeth.decode(paeth_encoded)
    
    codec.encode(paeth_decoded, outfile)
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
