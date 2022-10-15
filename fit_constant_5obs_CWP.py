#!/usr/bin/env python
# version: v3 (adapted for the Cartwheel paper, Salvaggio et al 2022)
# date: 2022/10/15
# author: mario <andrea.belfiore@inaf.it>
# name: fit_constant_5obs_CWP.py
# description:
#   Given a properly formatted table (see section below), describing 5 pairs of
#   measures, On & Off, for a number of sources, fit a constant flux model for
#   one of them.
#
#   This script computes the likelihood distribution for the net source counts
#   in each one of the 5 observations and converts it into a likelihood for
#   the source flux in each observation. Then, it searches for the common
#   value of the flux that maximises the combined likelihood. This operation
#   is first done with a brute force scan, then the global optimum is refined
#   down to the requested level of accuracy.
#   From the drop in TS = -2 * delta(logLike) caused by fixing the flux to
#   a constant value, we evaluate how significant is the variability.
#   Similarly, we measure the uncertainty in the best-fit flux.
#
#   Option "-c" specifies the coverage of uncertainties (default to 1 sigma).
#   Option "-s" allows to set the number of steps in the brute force scan.
#   Option "-a" specifies the relative accuracy on the stated flux values.
#   Option "-v" provides more statistics. Option "-q" suppresses most messages.
#
# example:
#     $> ./fit_constant_5obs_CWP.py -f counts_combined_v4.dat 3
#      src number: 3  "N11"
#      Obs "2019" Counts On=61 Off=30 bkg_ratio=33.5604 signif=17.8018
#      detected flux: 5.5739e-15 +/- 7.2210e-16
#      Obs "0200800101" Counts On=63 Off=377 bkg_ratio=8.4537 signif=2.3783
#      2 sigma upper limit flux: 6.7867e-15
#      Obs "0200800201" Counts On=79 Off=602 bkg_ratio=8.6317 signif=0.9802
#      2 sigma upper limit flux: 3.2932e-15
#      Obs "9531" Counts On=34 Off=26 bkg_ratio=33.5604 signif=12.5882
#      detected flux: 5.4661e-15 +/- 9.5400e-16
#      Obs "9807" Counts On=27 Off=26 bkg_ratio=33.5604 signif=10.8398
#      detected flux: 4.3193e-15 +/- 8.4981e-16
#     average flux: 4.6602e-15 -3.3721e-16 +4.1337e-16 erg/cm2/s
#     TS variability: 11.1492  Pval: 2.4937e-02  Z: 2.2424
#
# input data format:
#   Each row corresponds to a source from 1 up.
#   The columns of the table should be (this header is not necessary):
#     $> grep src_id counts_combined_v4.dat
#     # src_id  Non1    Noff1   zeta1   c2fl1       id_obs1 Non2    Noff2   zeta2   c2fl2       id_obs2     Non3    Noff3   zeta3   c2fl3       id_obs3     Non4    Noff4   zeta4   c2fl4       id_obs4 Non5    Noff5   zeta5   c2fl5       id_obs5 
#
#   This is an example of contents:
#     $> grep -v src_id counts_combined_v4.dat | nl
#     1 N6      17      31      35.1256 8.9168e-17  2019    41      377     8.4005  2.0046e-16  0200800101  89      602     8.6043  1.1960e-16  0200800201  3       34      35.1256 1.6000e-16  9531    4       25      35.1256 1.6337e-16  9807
#     2 N10     344     27      24.2077 8.9175e-17  2019    241     377     8.4270  1.9839e-16  0200800101  244     602     8.6385  1.2358e-16  0200800201  151     23      24.2077 1.5984e-16  9531    47      29      24.2077 1.7110e-16  9807
#     3 N11     61      30      33.5604 9.2266e-17  2019    63      377     8.4537  2.0004e-16  0200800101  79      602     8.6317  1.2076e-16  0200800201  34      26      33.5604 1.6302e-16  9531    27      26      33.5604 1.6280e-16  9807
#     ....
#   Here the first column (added through nl) is the src number, which has to be
#   passed to the script.
#
#   The only 20 columns actually used for computation are, for each observation
#   $obs=1,2,3,4,5 :    N_on_$obs    N_off_$obs    zeta_$obs    c2f_$obs
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
from scipy.stats import norm, chi2
from BLike import BLike # >=v7

def main():
  # read the command line and the input data
  Options = ParseCommandLine()
  n_steps = Options.steps
  if Options.verbose:
    chatter = 2
  elif Options.quiet:
    chatter = 0
  else:
    chatter = 1
  dtype = np.dtype([("srcid", "U12"),
      ("N_on_1", int), ("N_off_1", int), ("zeta_1", float), ("c2f_1", float), ("Obs_1", "U12"),
      ("N_on_2", int), ("N_off_2", int), ("zeta_2", float), ("c2f_2", float), ("Obs_2", "U12"),
      ("N_on_3", int), ("N_off_3", int), ("zeta_3", float), ("c2f_3", float), ("Obs_3", "U12"),
      ("N_on_4", int), ("N_off_4", int), ("zeta_4", float), ("c2f_4", float), ("Obs_4", "U12"),
      ("N_on_5", int), ("N_off_5", int), ("zeta_5", float), ("c2f_5", float), ("Obs_5", "U12")])
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

  # restructure the data
  N_on = [src["N_on_1"], src["N_on_2"], src["N_on_3"], src["N_on_4"], src["N_on_5"]]
  N_off = [src["N_off_1"], src["N_off_2"], src["N_off_3"], src["N_off_4"], src["N_off_5"]]
  zeta = [src["zeta_1"], src["zeta_2"], src["zeta_3"], src["zeta_4"], src["zeta_5"]]
  c2f = [src["c2f_1"], src["c2f_2"], src["c2f_3"], src["c2f_4"], src["c2f_5"]]
  Obs = [src["Obs_1"], src["Obs_2"], src["Obs_3"], src["Obs_4"], src["Obs_5"]]

  # compute the likelihood
  L = [BLike(x, y, z) for x, y, z in zip(zeta, N_on, N_off)] 
  flux_best = [Detect(l, x, o, Options.coverage, chatter) for l, x, o in zip(L, c2f, Obs)]
  TS_best = [l.TS(f/x) for l, f, x in zip(L, flux_best, c2f)]

  # find the optimum flux
  flux_min = np.min(flux_best)
  flux_max = np.max(flux_best)
  flux_opt = 0.
  TS_opt = sum([l.TS(flux_opt/x) for l, x in zip(L, c2f)])
  TS_min = sum([l.TS(flux_min/x) for l, x in zip(L, c2f)])
  TS_max = sum([l.TS(flux_max/x) for l, x in zip(L, c2f)])
  if chatter > 1:
    print("  flux min: %.6e  opt: %.6e  max: %.6e" % \
        (flux_min, flux_opt, flux_max))
    print("  TS min: %.2f  opt: %.2f  max: %.2f" % (TS_min, TS_opt, TS_max))
  for flux in np.linspace(flux_min, flux_max, n_steps):
    TS = sum([l.TS(flux/x) for l, x in zip(L, c2f)])
    if chatter > 1:
      print("    flux: %.6e  TS: %f = %f + %f + %f + %f + %f" % (flux, TS, \
          L[0].TS(flux/c2f[0]), L[1].TS(flux/sc2f[1]), L[2].TS(flux/c2f[2]), \
          L[3].TS(flux/c2f[3]), L[4].TS(flux/sc2f[4])))
    if TS < TS_opt:
      TS_opt = TS
      flux_opt = flux

  # refine the value
  dflux = (flux_max - flux_min) / n_steps
  flux_min = flux_opt - dflux
  flux_max = flux_opt + dflux
  TS_min = sum([l.TS(flux_min/x) for l, x in zip(L, c2f)])
  TS_max = sum([l.TS(flux_max/x) for l, x in zip(L, c2f)])
  flux_ave = (flux_max + flux_min) / 2.
  TS_ave = sum([l.TS(flux_ave/x) for l, x in zip(L, c2f)])
  if chatter > 1:
    print("  flux min: %.6e  ave: %.6e  max: %.6e" % \
        (flux_min, flux_ave, flux_max))
    print("  TS min: %.2f  ave: %.2f  max: %.2f" % (TS_min, TS_ave, TS_max))
  if flux_opt > 0:
    while (flux_max - flux_min) / flux_opt > Options.accuracy:
      flux_ave = (flux_max + flux_min) / 2.
      TS_ave = sum([l.TS(flux_ave/x) for l, x in zip(L, c2f)])
      if TS_min < TS_max:
        TS_max = TS_ave
        flux_max = flux_ave
      else:
        TS_min = TS_ave
        flux_min = flux_ave

  # compute the uncertainty on the average flux
  TS_unc = TS_ave + np.power(norm.isf((1 - Options.coverage)/2), 2)
  flux_min = 0.
  flux_max = 2. * np.max(flux_best)
  flux_lo = flux_max
  flux_hi = flux_min
  if chatter > 1:
    print("  flux min: %.6e  lo: %.6e  hi: %.6e  max: %.6e" % \
        (flux_min, flux_lo, flux_hi, flux_max))
    print("  TS opt: %.2f  ave: %.2f  1sig: %.2f" % (TS_opt, TS_ave, TS_unc))
  for flux in np.linspace(flux_min, flux_max, n_steps):
    TS = sum([l.TS(flux/x) for l, x in zip(L, c2f)])
    if chatter > 1:
      print("    flux: %.6e  TS: %f = %f + %f + %f + %f + %f" % (flux, TS, \
          L[0].TS(flux/c2f[0]), L[1].TS(flux/sc2f[1]), L[2].TS(flux/c2f[2]), \
          L[3].TS(flux/c2f[3]), L[4].TS(flux/sc2f[4])))
    if TS < TS_unc:
      if flux < flux_lo:
        flux_lo = flux
      if flux > flux_hi:
        flux_hi = flux
  if chatter > 1:
    print("  flux min: %.6e  lo: %.6e  hi: %.6e  max: %.6e" % \
        (flux_min, flux_lo, flux_hi, flux_max))

  # refine the uncertainty
  for flux in np.linspace(flux_lo - dflux, flux_lo + dflux, n_steps):
    TS = sum([l.TS(flux/x) for l, x in zip(L, c2f)])
    if TS < TS_unc:
      if flux < flux_lo:
        flux_lo = flux
  for flux in np.linspace(flux_hi - dflux, flux_hi + dflux, n_steps):
    TS = sum([l.TS(flux/x) for l, x in zip(L, c2f)])
    if TS < TS_unc:
      if flux > flux_hi:
        flux_hi = flux

  # compute and print out the results
  dTS = TS_ave - sum(TS_best)
  Pval = chi2.sf(float(dTS), len(TS_best) - 1)
  Zscore = norm.isf(Pval/2)
  print("average flux: %.4e -%.4e +%.4e erg/cm2/s" %
    (flux_ave, flux_ave - flux_lo, flux_hi - flux_ave))
  print("TS variability: %.4f  Pval: %.4e  Z: %.4f" % (dTS, Pval, Zscore))

def Detect(my_L, my_c2f, obs_id, coverage, chatter):
  flux_best = my_L.MostLikely() * my_c2f
  if chatter > 0:
    print(" Obs \"%s\" Counts On=%d Off=%d bkg_ratio=%.4f signif=%.4f" %
        (obs_id, my_L.N_on, my_L.N_off, my_L.bkg_ratio, my_L.Significance()))
    if my_L.Significance() > 3.:
      cts_lo, cts_hi = my_L.HDI(coverage)
      flux = .5 * (cts_hi + cts_lo) * my_c2f
      dflux = .5 * (cts_hi - cts_lo) * my_c2f
      print(" detected flux: %.4e +/- %.4e" % (flux, dflux))
    else:
      upper_limit = my_L.UpperLimit() * my_c2f
      print(" 2 sigma upper limit flux: %.4e" % (upper_limit))
  return flux_best

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
  Parser.add_argument('--steps', '-s', type=int,\
      default=100, help='Number of steps in preliminary scan [100]')
  Parser.add_argument('--accuracy', '-a', type=float,\
      default=1e-4, help='Relative accuracy of average [1e-4]')
  Parser.add_argument('--coverage', '-c', type=float,\
      default=0.6826895, help='Coverage of uncertainties [0.6826895]')
  Parser.add_argument('--verbose', '-v', action='store_true', \
      default=False, help='Extended output')
  Parser.add_argument('--quiet', '-q', action='store_true', \
      default=False, help='Reduced output')
  return Parser.parse_args()

if __name__ == '__main__':
  main()
