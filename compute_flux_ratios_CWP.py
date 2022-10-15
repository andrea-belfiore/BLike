#!/usr/bin/env python
# version: v3 (adapted for the Cartwheel paper, Salvaggio et al 2022)
# date: 2022/10/15
# author: mario <andrea.belfiore@inaf.it>
# name: compute_flux_ratios_CWP.py
# description:
#   Given a properly formatted table (see section below), describing 3 pairs of
#   measures, On & Off, for a number of sources, extract the maximum flux ratio
#   for one of them.
#
#   This script takes each combination of 2 out of 3 observations and focuses
#   on a single source (therefore each observation is reduced to a pair of
#   measures, On & Off). Blike draws from the posterior distribution of the net
#   source counts, converted into distribution of flux, for each observation.
#   By sampling from these distributions we bootstrap a distribution for their
#   flux ratio, that we summarize through the highest density interval with
#   the requested coverage.
#   We choose and report the 2 observations that maximize this ratio.
#
#   Option "-c" specifies the coverage of uncertainties (default to 1 sigma).
#   Option "-r" allows to set the number of points to be used in the bootstrap.
#   Option "-v" provides more statistics. Option "-q" suppresses most messages.
#
#   Is this source variable? It looks like but we cannot really tell as we do
#   not know how the flux ratio behaves for a similar constant source.
#
# example:
#     $> ./compute_flux_ratios_CWP.py -f counts_v4.dat -r 10000  3
#      src number: 3  "N3"
#      bootstrap the flux ratio from 10000 realisations
#      (flux in obs 1) / (flux in obs 2) = 0.4385 +/- 0.0794
#      (flux in obs 1) / (flux in obs 3) = 0.5848 +/- 0.1176
#      (flux in obs 2) / (flux in obs 1) = 2.2056 +/- 0.3968
#      (flux in obs 2) / (flux in obs 3) = 1.3204 +/- 0.2482
#      (flux in obs 3) / (flux in obs 1) = 1.6207 +/- 0.3199
#      (flux in obs 3) / (flux in obs 2) = 0.7335 +/- 0.1358
#      Largest ratio:
#     (flux in obs 2) / (flux in obs 1) = 2.2056 +/- 0.3968
#
#   For weak sources the posterior distribution of their flux ratio is far
#   from symmetric, so you might want to use the "-v" option to display
#   asymmetric error bars (computed in various ways).
# input data format:
#   Each row corresponds to a source from 1 up.
#   The columns of the table should be (this header is not necessary):
#     $> grep src_id counts_v4.dat
#     # src_id  Non1    Noff1   zeta1   c2fl1       id_obs1 Non2    Noff2   zeta2   c2fl2       id_obs2 Non3    Noff3   zeta3   c2fl3       id_obs3 
#
#   This is an example of contents:
#     $> grep -v src_id counts_v4.dat | nl
#     1 N1      58      15      12.7922 8.9437e-17  2019    21      11      12.7922 1.7072e-16  9531    12      11      12.7922 1.6305e-16  9807
#     2 N2      35      2       4.7922  9.7765e-17  2019    22      4       4.7922  1.6175e-16  9531    15      3       4.7922  1.6278e-16  9807
#     3 N3      55      19      14.6956 9.3069e-17  2019    70      16      14.6956 1.6175e-16  9531    52      19      14.6956 1.6250e-16  9807
#     ....
#   Here the first column (added through nl) is the src number, which has to be
#   passed to the script.
#
#   The only 12 columns actually used for computation are, for each observation
#   $obs=1,2,3 :    N_on_$obs    N_off_$obs    zeta_$obs    c2f_$obs
#   The first 3 parameters are the standard BLike input (N_on the number of
#   counts in the On measure; N_off those in the Off measure; zeta the ratio of
#   expected background counts Off/On. To first order, zeta is the ratio of the
#   areas of the respective sky regions).
#   The 4th parameter, obtained with XSpec, is the conversion factor from
#   counts to flux (as we are comparing flux, but it could as well be rate
#   or luminosity as long as all the values are consistent: the ratio doesn't
#   change).
#   The 5th parameter is used to identify each observation in the output.
#

import sys, argparse
import numpy as np
from BLike import BLike # >=v7

def main():
  # read the command line and the input data
  Options = ParseCommandLine()
  n_real = Options.realizations
  if Options.verbose:
    chatter = 2
  elif Options.quiet:
    chatter = 0
  else:
    chatter = 1
  dtype = np.dtype([("srcid", "U12"),
      ("N_on_1", int), ("N_off_1", int), ("zeta_1", float), ("c2f_1", float), ("Obs_1", "U12"),
      ("N_on_2", int), ("N_off_2", int), ("zeta_2", float), ("c2f_2", float), ("Obs_2", "U12"),
      ("N_on_3", int), ("N_off_3", int), ("zeta_3", float), ("c2f_3", float), ("Obs_3", "U12")])
  try:
    data = np.loadtxt(Options.file, dtype=dtype)
  except:
    print("Cannot read the input file \"%s\"" % (Options.file.name))
    sys.exit(1)
  if len(data) < Options.src_num:
    print("The input file contains data only for %d sources" % (len(data)))
    sys.exit(2)
  src = data[Options.src_num - 1]
  if chatter > 0:
    print(" src number: %d  \"%s\"" % (Options.src_num, src["srcid"]))
    print(" bootstrap the flux ratio from %d realisations" % (n_real))

  # restructure the data
  N_on = [src["N_on_1"], src["N_on_2"], src["N_on_3"]]
  N_off = [src["N_off_1"], src["N_off_2"], src["N_off_3"]]
  zeta = [src["zeta_1"], src["zeta_2"], src["zeta_3"]]
  c2f = [src["c2f_1"], src["c2f_2"], src["c2f_3"]]
  Obs = [src["Obs_1"], src["Obs_2"], src["Obs_3"]]

  # compute the likelihood
  L = [BLike(x, y, z) for x, y, z in zip(zeta, N_on, N_off)] 

  # get the posterior flux distribution for the 3 observations
  flx = np.zeros_like(L)
  for i in range(len(L)):
    # this generates n_real number drawn from the posterior distribution
    cts = [L[i].Quantile(X) for X in np.random.uniform(size=n_real)]
    if chatter > 1:
      DisplayStats(L[i], cts, Obs[i], i + 1)
    flx[i] = np.multiply(cts, c2f[i])

  # get the distribution of flux ratios for each pair of observations
  max_ratio = 1.
  max_ratio_unc = 0.
  obs_A = 0
  obs_B = 0
  for i in range(len(L)):
    for j in range(len(L)):
      if i == j:
        continue 
      if chatter > 1:
        print("  Compare obs %d  \"%s\" and obs %d  \"%s\":" % \
            (i + 1, Obs[i], j + 1, Obs[j]))
      ratio, ratio_unc = CharacteriseRatio(flx[i], flx[j], chatter, Options.coverage)
      if chatter == 1:
        print(" (flux in obs %d) / (flux in obs %d) = %.4f +/- %.4f" % \
            (i + 1, j + 1, ratio, ratio_unc))
      if ratio > max_ratio:
        max_ratio = ratio
        max_ratio_unc = ratio_unc
        obs_A = i + 1
        obs_B = j + 1

  # write out the results
  if chatter > 0:
    print(" Largest ratio:")
  print("(flux in obs %d) / (flux in obs %d) = %.4f +/- %.4f" % \
      (obs_A, obs_B, max_ratio, max_ratio_unc))

def CharacteriseRatio(flx_A, flx_B, chatter, coverage=.682689492137, \
    accuracy=1e-3):
  ratio = np.divide(flx_A, flx_B)
  hi_val = np.quantile(ratio, .5 + coverage / 2.)
  lo_val = np.quantile(ratio, .5 - coverage / 2.)
  sym_med = (hi_val + lo_val) / 2.
  sym_size = hi_val - lo_val
  # find the narrowest interval at fixed coverage
  min_size = sym_size
  min_med = sym_med
  for lc in np.arange(0, 1 - coverage, accuracy):
    hc = lc + coverage
    hv = np.quantile(ratio, hc)
    lv = np.quantile(ratio, lc)
    if hv - lv < min_size:
      min_size = hv - lv
      min_med = (hv + lv) / 2.
  if chatter > 1:
    ave_A = np.average(flx_A)
    ave_B = np.average(flx_B)
    std_A = np.std(flx_A)
    std_B = np.std(flx_B)
    rat_AB = ave_A / ave_B
    unc_rat_AB = rat_AB * np.hypot(std_A/ave_A, std_B/ave_B)
    print("  ave[A]/ave[B]=%.4f unc=%.4f med[A]/med[B]=%.4f" %
        (rat_AB, unc_rat_AB, np.median(flx_A)/np.median(flx_B)))
    print("  ave=%.4f std=%.4f med=%.4f" %
        (np.average(ratio), np.std(ratio), np.median(ratio)))
    print("  med(symmetric)=%.4f unc(symmetric)=%.4f" %
        (sym_med, sym_size / 2.))
    print("  med(narrowest)=%.4f unc(narrowest)=%.4f" %
        (min_med, min_size / 2.))
    print("")
  return min_med, min_size / 2.

def DisplayStats(my_L, cts, obs_id, idx):
  print("  Observation %d  \"%s\":" % (idx, obs_id))
  print("  Input data:")
  print("    N_on=%d N_off=%d zeta=%d (N_src=%.2f) det=%.2f sigma" % 
      (my_L.N_on, my_L.N_off, my_L.bkg_ratio, \
      my_L.N_on - my_L.N_off/my_L.bkg_ratio, my_L.Significance()))
  print("  Posterior properties:")
  print("    MostLikely=%.4f +%.4f -%.4f Median=%.4f HDI=[%.4f, %.4f]" %
      (my_L.MostLikely(), my_L.ErrorBar(), my_L.ErrorBar(up=False), \
      my_L.Quantile(.5), my_L.HDI()[0], my_L.HDI()[1]))
  print("  Sample properties:")
  print("    N_real=%d ave=%.4f std=%.4f med=%.4f" %
      (len(cts), np.average(cts), np.std(cts), np.median(cts)))
  print("")


def ParseCommandLine():
  """ParseCommandLine()"""
  """Read the arguments passed on the command line"""
  """to the script."""
  Parser = argparse.ArgumentParser(
      description='Compute the flux ratios for 3 On-Off pairs of measures')
  Parser.add_argument('src_num', type=int,\
      help='Source number')
  Parser.add_argument('--file', '-f', type=argparse.FileType('r'),\
      default='counts.dat', help='Input file [counts.dat]')
  Parser.add_argument('--realizations', '-r', type=int,\
      default=1000, help='Number of realizations [1000]')
  Parser.add_argument('--coverage', '-c', type=float,\
      default=0.6826895, help='Coverage of uncertainties [0.6826895]')
  Parser.add_argument('--verbose', '-v', action='store_true', \
      default=False, help='Extended output')
  Parser.add_argument('--quiet', '-q', action='store_true', \
      default=False, help='Reduced output')
  return Parser.parse_args()

if __name__ == '__main__':
  main()
