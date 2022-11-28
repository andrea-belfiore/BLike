# BLike and the variability of X-ray sources in the Cartwheel galaxy

Here you can find a few python scripts that use BLike to test
for and characterise the variability of faint X-ray sources.

You can use them to reproduce the results of table 4 in our
paper: "The largest bright ULX population in a galaxy: X-ray
variability and Luminosity Function in the Cartwheel ring Galaxy"
(Salvaggio et al 2022).

BLike is a python module that implements a class, BLike, that
describes an On-Off pair of measurements. This is a typical
case in counting experiments and in particular, in high-energy
astronomy: you have an observation On-source, contaminated
by background, and one Off-source, free of the source. In our
case the two measures are taken at the same time, in different
regions of the sky.

While proper documentation is still lacking, this library has
reached quite a stable and reliable level. A paper is under way.
Hopefully, I will add some notes here before too long.

If we assume a flat prior for both background and source net
expected counts and Poisson processes for both, the posterior
distribution (in this case the likelihood function, normalised)
has an analytical closed expression. Therefore, in many cases
you can work out estimates and tests without MCMC or other
approximations. In one case, however, we show how to draw
random values from the posterior distribution to work out a
derived quantity (the ratio of 2 fluxes). 

Feel free to use it, maybe adapting the scripts to your needs.
Please, let me know what you think of it, if you find any bug,
and whether you want me to add any feature.

First thing in my TODO list is to implement Jeffreys' prior
and maybe some other prior as well. Looking for variability
I believe I'm free from Malmquist's bias, but in many other
applications it becomes essential to deal with it. 
