#!/usr/bin/env python
# version: v4 (adapted for the Cartwheel paper, Salvaggio et al 2022)
# date: 2022/10/22
# author: mario <andrea.belfiore@inaf.it>
# name: test_variability_Zezas_CWP.py
# description:
#   Given a properly formatted table (see section below), describing 3 pairs of
#   measures, On & Off, for a number of sources, test for variability following
#   the method of Zezas et al 2006 (adapted to the low-counts regime) for one
#   of them.
#
#   This script takes each combination of 2 out of 3 observations and focuses
#   on a single source (therefore each observation is reduced to a pair of
#   measures, On & Off). Blike computes the posterior distribution of the net
#   source counts, converted into distribution of flux, for each observation.
#   Then each pair of observations is compared against a constant flux model.
#   From the drop in TS = -2 * delta(logLike) caused by fixing the flux to
#   a constant value, we evaluate how significant is the variability.
#   We choose and report the 2 observations that maximize this value, after
#   a correction for the number of trials (3 in this case).
#
#   Option "-c" specifies the coverage of uncertainties (default to 1 sigma).
#   Option "-s" allows to set the number of steps in the brute force scan.
#   Option "-a" specifies the relative accuracy on the stated flux values.
#   Option "-v" provides more statistics. Option "-q" suppresses most messages.
#
# example:
#     $> ./test_variability_Zezas_CWP.py -f counts_v4.dat 3
#      src number: 3  "N3"
#      Obs "2019" Counts On=55 Off=19 bkg_ratio=14.6956 signif=14.8013
#      detected flux: 5.0234e-15 +/- 6.9220e-16
#      Obs "9531" Counts On=70 Off=16 bkg_ratio=14.6956 signif=17.4020
#      detected flux: 1.1190e-14 +/- 1.3563e-15
#      Obs "9807" Counts On=52 Off=19 bkg_ratio=14.6956 signif=14.2962
#      detected flux: 8.2838e-15 +/- 1.1754e-15
#      variability between obs 1 and obs 2: TS = 19.1289  Z = 4.3737
#      variability between obs 1 and obs 3: TS = 6.2405  Z = 2.4981
#      variability between obs 2 and obs 3: TS = 2.6402  Z = 1.6249
#     max variability: TS = 19.1289  Z = 4.1276
#
#   Mind the difference in the TS to Z-score (N sigma) conversion, due to the
#   number of trials: TS = 19.1289 corresponds 4.3 sigma in single trial, but
#   it becomes 4.1 sigma after trial correction.
#
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
from scipy.stats import norm, chi2
from BLike import BLike # >=v8

def main():
  # read the command line and the input data
  Options = ParseCommandLine()
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

  # restructure the data
  N_on = [src["N_on_1"], src["N_on_2"], src["N_on_3"]]
  N_off = [src["N_off_1"], src["N_off_2"], src["N_off_3"]]
  zeta = [src["zeta_1"], src["zeta_2"], src["zeta_3"]]
  c2f = [src["c2f_1"], src["c2f_2"], src["c2f_3"]]
  Obs = [src["Obs_1"], src["Obs_2"], src["Obs_3"]]

  # compute the likelihood
  L = [BLike(x, y, z) for x, y, z in zip(zeta, N_on, N_off)] 
  flux_best = [Detect(l, x, o, Options.coverage, chatter) for l, x, o in zip(L, c2f, Obs)]

  fmt = " variability between obs %d and obs %d: TS = %.4f  Z = %.4f"
  TS_max = 0
  for i in range(len(L)):
    for j in range(i + 1, len(L)):
      TS = compare(L[i], L[j], c2f[i], c2f[j], \
          Options.steps, Options.accuracy, chatter)
      if chatter > 0:
        print(fmt % (i + 1, j + 1, TS, TS2Z(TS, 1)))
      if TS > TS_max:
        TS_max = TS

  # write out the results
  print("max variability: TS = %.4f  Z = %.4f" % (TS_max, TS2Z(TS_max, 3)))

def compare(L_a, L_b, c2f_a, c2f_b, n_steps, accuracy, chatter):
  flux_best_a = L_a.MostLikely() * c2f_a
  flux_best_b = L_b.MostLikely() * c2f_b

  # find the optimum flux
  flux_min = np.min([flux_best_a, flux_best_b])
  flux_max = np.max([flux_best_a, flux_best_b])
  flux_opt = 0.
  TS_opt = L_a.TS(0.) + L_b.TS(0.)
  TS_min = L_a.TS(flux_min/c2f_a) + L_b.TS(flux_min/c2f_b)
  TS_max = L_a.TS(flux_max/c2f_a) + L_b.TS(flux_max/c2f_b)
  if chatter > 1:
    print("  flux min: %.6e  opt: %.6e  max: %.6e" % \
        (flux_min, flux_opt, flux_max))
    print("  TS min: %.2f  opt: %.2f  max: %.2f" % (TS_min, TS_opt, TS_max))
  for flux in np.linspace(flux_min, flux_max, n_steps):
    TS = L_a.TS(flux/c2f_a) + L_b.TS(flux/c2f_b)
    if chatter > 1:
      print("    flux: %.6e  TS: %f = %f + %f" % \
          (flux, TS, L_a.TS(flux/c2f_a), L_b.TS(flux/c2f_b)))
    if TS > TS_opt:
      TS_opt = TS
      flux_opt = flux

  # refine the value
  dflux = (flux_max - flux_min) / n_steps
  flux_min = flux_opt - dflux
  flux_max = flux_opt + dflux
  TS_min = L_a.TS(flux_min/c2f_a) + L_b.TS(flux_min/c2f_b)
  TS_max = L_a.TS(flux_max/c2f_a) + L_b.TS(flux_max/c2f_b)
  flux_ave = (flux_max + flux_min) / 2.
  TS_ave = L_a.TS(flux_ave/c2f_a) + L_b.TS(flux_ave/c2f_b)
  if chatter > 1:
    print("  flux min: %.6e  ave: %.6e  max: %.6e" % \
        (flux_min, flux_ave, flux_max))
    print("  TS min: %.2f  ave: %.2f  max: %.2f" % (TS_min, TS_ave, TS_max))
  if flux_opt > 0:
    while (flux_max - flux_min) / flux_opt > accuracy:
      flux_ave = (flux_max + flux_min) / 2.
      TS_ave = L_a.TS(flux_ave/c2f_a) + L_b.TS(flux_ave/c2f_b)
      if TS_min > TS_max:
        TS_max = TS_ave
        flux_max = flux_ave
      else:
        TS_min = TS_ave
        flux_min = flux_ave
  TS_best_a = L_a.TS(L_a.MostLikely())
  TS_best_b = L_b.TS(L_b.MostLikely())
  dTS = (TS_best_a + TS_best_b) - TS_ave
  return dTS

def TS2Z(TS, Ntrial):
  Pval_single_trial = 2. * norm.sf(np.sqrt(np.double(TS)))
  if TS < 25:
    Pval = 1 - np.power(1. - Pval_single_trial, Ntrial)
  else:
    Pval = Ntrial * Pval_single_trial
  Z = norm.isf(Pval/2)
  return Z

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
