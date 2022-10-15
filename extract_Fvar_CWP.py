#!/usr/bin/env python
# version: v1 (adapted for the Cartwheel paper, Salvaggio et al 2022)
# date: 2022/10/15
# author: mario <andrea.belfiore@inaf.it>
# name: extract_Fvar_CWP.py
# description:
#   Given a properly formatted table (see section below), describing 3 pairs of
#   measures, On & Off, for a number of sources, extract the fractional
#   variability amplitude (Fvar) as described in Edelson et al 2002 (and here
#   adapted to the low-counts regime) for one of them.
#
#   Is this source variable? It looks like but we cannot really tell as we do
#   not know how Fvar behaves for a similar constant source. In principle we
#   could simulate such a source and see how the Fvar distributes. We have not
#   done it yet.
#   However, the goal of Fvar is to quantify the relative average spread of the
#   flux values, rather than testing the constancy of a source.
#
#   Option "-c" specifies the coverage of uncertainties (default to 1 sigma).
#   Option "-v" provides more statistics. Option "-q" suppresses most messages.
#
# example:
#     $> ./extract_Fvar_CWP.py -f counts_v4.dat 3
#      src number: 3  "N3"
#      Obs "2019" Counts On=55 Off=19 bkg_ratio=14.6956 signif=14.8013
#      detected flux: 5.0234e-15 +/- 6.9220e-16
#      Obs "9531" Counts On=70 Off=16 bkg_ratio=14.6956 signif=17.4020
#      detected flux: 1.1190e-14 +/- 1.3563e-15
#      Obs "9807" Counts On=52 Off=19 bkg_ratio=14.6956 signif=14.2962
#      detected flux: 8.2838e-15 +/- 1.1754e-15
#      flux var = 9.5173e-30  >  flux UMS = 1.2334e-30
#     Fvar = 0.3525 +/- 0.0814
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
from BLike import BLike # >=v7

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
  # get the center and half-size of the HDI with the right coverage
  flux = np.zeros_like(L)
  flux_unc = np.zeros_like(L)
  for i in range(len(L)):
    cts_lo, cts_hi = L[i].HDI(Options.coverage)
    flux[i] = 0.5 * (cts_hi + cts_lo) * c2f[i]
    flux_unc[i] = 0.5 * (cts_hi - cts_lo) * c2f[i]

  flux_ave = np.mean(flux)
  flux_var = 3./2. * np.var(flux)
  flux_ums = np.mean(np.power(flux_unc, 2))
  if flux_var < flux_ums:
    if chatter > 0:
      print(" flux var = %.4e  <  flux UMS = %.4e" % (flux_var, flux_ums))
    print("Fvar = 0")
    return

  Fvar = np.sqrt(flux_var - flux_ums) / flux_ave
  UMS = flux_ums / np.power(flux_ave, 2)
  Fvar_err = np.sqrt(UMS / 3. * (1. + 0.5 * UMS / np.power(Fvar, 2)))
  if chatter > 0:
    print(" flux var = %.4e  >  flux UMS = %.4e" % (flux_var, flux_ums))
  if chatter <= 1:
    print("Fvar = %.4f +/- %.4f" % (Fvar, Fvar_err))
  if chatter > 1:
    str1 = "(normalising over the average of the HDI midpoint flux)"
    print("Fvar = %.4f +/- %.4f  %s" % (Fvar, Fvar_err, str1))
    flux_ave2 = np.mean(flux_best)
    Fvar2 = np.sqrt(flux_var - flux_ums) / flux_ave2
    UMS2 = flux_ums / np.power(flux_ave2, 2)
    Fvar2_err = np.sqrt(UMS2 / 3. * (1. + 0.5 * UMS2 / np.power(Fvar2, 2)))
    str2 = "(normalising over the average of best-fit flux)"
    print("  Fvar2 = %.4f +/- %.4f  %s" % (Fvar2, Fvar2_err, str2))

    # find the optimum flux
    n_steps = 1000
    flux_min = np.min(flux_best)
    flux_max = np.max(flux_best)
    flux_opt = 0.
    TS_opt = sum([l.TS(flux_opt/x) for l, x in zip(L, c2f)])
    for f in np.linspace(flux_min, flux_max, n_steps):
      TS = sum([l.TS(f/x) for l, x in zip(L, c2f)])
      if TS < TS_opt:
        TS_opt = TS
        flux_opt = f
    # refine the value
    dflux = (flux_max - flux_min) / n_steps
    flux_min = flux_opt - dflux
    flux_max = flux_opt + dflux
    TS_min = sum([l.TS(flux_min/x) for l, x in zip(L, c2f)])
    TS_max = sum([l.TS(flux_max/x) for l, x in zip(L, c2f)])
    flux_ave3 = (flux_max + flux_min) / 2.
    TS_ave = sum([l.TS(flux_ave3/x) for l, x in zip(L, c2f)])
    if flux_opt > 0:
      while (flux_max - flux_min) / flux_opt > 1e-4:
        flux_ave3 = (flux_max + flux_min) / 2.
        TS_ave = sum([l.TS(flux_ave3/x) for l, x in zip(L, c2f)])
        if TS_min < TS_max:
          TS_max = TS_ave
          flux_max = flux_ave3
        else:
          TS_min = TS_ave
          flux_min = flux_ave3
    Fvar3 = np.sqrt(flux_var - flux_ums) / flux_ave3
    UMS3 = flux_ums / np.power(flux_ave3, 2)
    Fvar3_err = np.sqrt(UMS3 / 3. * (1. + 0.5 * UMS3 / np.power(Fvar3, 2)))
    str3 = "(normalising over the overall best-fit flux)"
    print("  Fvar3 = %.4f +/- %.4f  %s" % (Fvar3, Fvar3_err, str3))


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
      description='Extract the Fvar from 3 On-Off pairs of measures')
  Parser.add_argument('src_num', type=int,\
      help='Source number')
  Parser.add_argument('--file', '-f', type=argparse.FileType('r'),\
      default='counts.dat', help='Input file [counts.dat]')
  Parser.add_argument('--coverage', '-c', type=float,\
      default=0.6826895, help='Coverage of uncertainties [0.6826895]')
  Parser.add_argument('--verbose', '-v', action='store_true', \
      default=False, help='Extended output')
  Parser.add_argument('--quiet', '-q', action='store_true', \
      default=False, help='Reduced output')
  return Parser.parse_args()

if __name__ == '__main__':
  main()
