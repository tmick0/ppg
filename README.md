# PPG
Python Pixel Glitcher -- Disturbs image data to create cool effects

## Overview

PPG mostly consists of a set of filters for buliding image-glitching pipelines.
It consists of a C module which implements the actual filters, and some Python
bindings that allow them to easily be stitched together.

While image glitching is usually performed by manipulating the compressed binary
data of an image file, PPG takes a different approach. PPG applies transforms to
actual decoded images, then introduces errors to the transformed data before
reversing the transform to obtain an altered image. Some more information about
the philosophy behind PPG can be found at:
http://lo.calho.st/projects/adventures-in-image-glitching/

## Building the C module

A Makefile is provided for convenience. If you have the necessary Python
development packages, then the Makefile should take care of everything else for
you and produce the required ppgfilter.so.

## Using the Python module

The dependencies of the Python module are listed in requirements.txt. It is
written for Python 2.7 but might also work in Python 3 with a few changes...

Your main interface to PPG's functionality is through the "filter" module. All
of the image transformations are implemented here.

PPG's filters operate on images represented as Numpy arrays. An image is
essentially a three-dimensional array -- two dimensions are the dimensions of
the actual image, and the third separates the R/G/B (and sometimes A) color
channels. A convenience module, "codec," is provided to allow images to be
imported in this format (and exported from it).

To create glitch effects in an image, you must combine a transforming filter
with a corrupting filter. Currently, the following transforming filters are
supported:

 - SubFilter
 - UpFilter
 - PaethFilter
 - BrokenPaethFilter
 - AverageFilter
 - BrokenAverageFilter
 - RandomLineFilter

And a single corrupting filter, the UniformGlitchFilter.

Each of these filters has its own documentation if you want to learn about how
they work.

To combine filters and obtain interesting effects, you can utilize the
FilterStack and FilterChain classes; these act the same as any other filter,
and you can actually combine them arbitrarily. FilterStack and FilterChain
differ in their orderings of the encode (transform) and decode (reverse
transform) operations. A FilterChain will execute each filter in the chain by
running encode, then running decode consecutively. However, a FilterStack will
run the encode operations for all of the filters it contains, then run the
decode operations in the reverse order.

FilterStacks are necessary to allow the effects of a corrupting filter to
propagate; first, a transform must be applied, then the corruption, then a 
reverse transform -- the FilterStack facilitates this. On the other hand,
FilterChains allow additional layers of glitches to be added.

For an example of the usage of FilterChains and FilterStacks, take a look at
example.py.

## Using the example script

The example script contains a filter chain suitable for producing decent
glitch effects on most images. It takes four command line parameters: input
filename, output filename, glitch multiplier (optional), and quiet (optional).
The glitch multiplier is a float scalar. The quiet param will suppress the
performance output if its value is "quiet" or "q".

For example (default behavior):

     $ python example.py input.jpg output.jpg
     [+] FilterChain starting
        [+] FilterStack starting
            [+] Encoded with RandomLineFilter in 0.003s.
            [+] Encoded with UniformGlitchFilter in 0.001s.
            [-] Decoded with UniformGlitchFilter in 0.000s.
            [-] Decoded with RandomLineFilter in 0.002s.
        [+] Encoded with FilterStack in 0.006s.
        [-] Decoded with FilterStack in 0.000s.
        [+] FilterStack starting
            [+] Encoded with BrokenPaethFilter in 0.007s.
            [+] Encoded with UniformGlitchFilter in 0.002s.
            [-] Decoded with UniformGlitchFilter in 0.000s.
            [-] Decoded with BrokenPaethFilter in 0.008s.
        [+] Encoded with FilterStack in 0.016s.
        [-] Decoded with FilterStack in 0.000s.
        [+] FilterStack starting
            [+] Encoded with BrokenAverageFilter in 0.002s.
            [+] Encoded with UniformGlitchFilter in 0.001s.
            [-] Decoded with UniformGlitchFilter in 0.000s.
            [-] Decoded with BrokenAverageFilter in 0.002s.
        [+] Encoded with FilterStack in 0.005s.
        [-] Decoded with FilterStack in 0.000s.
    [+] FilterChain finished in 0.028s.

     
     
For example (50% glitchiness, quiet):

     $ python example.py input.jpg output.jpg 0.5 quiet
     
For example (200% glitchiness, quiet):

     $ python example.py input.jpg output.jpg 2.0 quiet

## Documentation

Docstrings are included for this entire package. There is also a convenience
method in the Makefile to compile the documentation: make docs. The convenience
method requires pdoc to be installed.
