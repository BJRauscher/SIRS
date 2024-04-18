# Simple Improved Reference Subtraction Examples

Bernard.J.Rauscher@nasa.gov<br>
NASA Goddard Space Flight Center

Simple Improved Reference Subtraction (SIRS) uses training data to compute frequency dependent weights that can be used to make reference corretions using an HxRG's reference columns. The training data are required to be a large set of up-the-ramp sampled darks.

SIRS consists of both a "front end" that computes the weights and a "back end" that applies them. Although the front end computation is currently available only in Julia, if the weights are known, there is a python-3 back end.
