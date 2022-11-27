# version: v8
# date: 2022/10/22
# author: mario <andrea.belfiore@inaf.it>
# name: BLike.py
# description:
#   Library for computing significance, estimates, uncertainties, and
#   upper limits in the low counts regime.
# todo:
#   - add Jeffreys' prior for both signal and background
# changelog:
#   v0 - basic implementation of all the methods
#   v1 - new algorithm for the U coefficients and the case N_on=0
#   v2 - made some methods and attributes private + debugged
#   v3 - isolate the Pavl to Z conversion and the polynomial class
#   v4 - new coefficients for large N_off
#   v5 - some explicit casting required by recent python
#   v6 - internals as float128 to manage larger counts
#   v7 - highest density interval (HDI) and other methods
#   v8 - profile likelihood and TS

## by now, let's assume the user has numpy and scipy installed
import numpy as np # polyval exp log
import scipy.stats as st # binom norm
import scipy.special as sp # comb

# this is a global setting, in order to bypass crazy requests
best_accuracy = 1.e-7
# these are only for upper limits and error bars
min_coverage = 1. - st.norm.sf(.5)*2. # .5 sigma (.382925)
max_coverage = 1. - st.norm.sf(3.5)*2. # 3.5 sigma (.999535)
debug = False

def PvalToZscore(X):
  if X >= .5:
    return 0.
  if X < 9.97e-316:
    return 38.
  # the stat version is not precise and overflows quickly
  # st.norm.ppf(1. - X / 2.)
  # credit Sergei Winitski 2008 eq.7
  a = (3 * (4 - np.pi)) / (4. * (np.pi - 3.))
  Y = np.log(4 * X * (1 - X)) / 2.

  return np.sqrt(2) * np.sqrt(np.sqrt((a + Y)**2 - a * Y * np.pi) - (a + Y))

def Polynomial(coeff, x):
  val = 0.
  for c in coeff[::-1]:
    val = val * x + c
  #print np.polyval(coeff[::-1], x) , val, coeff, x
  return val

class BLike:
  """A class for Bayesian Likelihood computation in low-counts regime"""
  def __init__(self, bkg_ratio=1., N_on=0, N_off=0):
    """bkg_ratio is the ratio between the expected background counts"""
    """in the Off measure over those in the On measure."""
    """This ratio generally depends on the effective area, aperture,"""
    """exposure, and detailed modeling of the background component."""
    """N_on and N_off are the observed counts in the two measures."""
    # store the basic params
    self.bkg_ratio = bkg_ratio
    self.N_on = int(N_on)
    self.N_off = int(N_off)

    # coefficients of the maximum profile likelihood
    self.__Am = (self.N_off - self.N_on) / (self.bkg_ratio + 1.)
    self.__Ap = (self.N_off + self.N_on) / (self.bkg_ratio + 1.)
    if self.N_off < self.bkg_ratio * self.N_on:
      self.__base_off = self.N_off / self.bkg_ratio
      self.__base_on = self.N_on
    else:
      self.__base_off = self.__Ap
      self.__base_on = self.__Ap

    # polynomial coefficients of the Likelihood (V)
    # of its derivative (W) and its integral (U)
    self.__W = np.zeros(self.N_on + 1, dtype=np.float128)
    self.__V = np.zeros(self.N_on + 2, dtype=np.float128)
    self.__U = np.zeros(self.N_on + 2, dtype=np.float128)
    # cache some useful quantities for efficiency
    self.__norm = np.float128(0.)
    self.__powr = np.zeros(self.N_on + 1, dtype=np.float128)
    self.__fact = np.zeros(self.N_on + 1, dtype=np.float128)
    self.__bincoeff = np.zeros(self.N_on + 1, dtype=np.float128)

    # make room for caching values
    self.__arg_max = 0.
    self.__accuracy = .1
    self.__max_like = -1.

  def __PrepareCache(self):
    """Store the first N_on+N_off factorials and other terms for efficiency"""
    self.__fact[0] = 1.
    for j in range(1, self.N_on + 1):
      self.__fact[j] = self.__fact[j-1] * np.double(j)
      if debug:
         print('__fact[%d] = %.4f' % (j, self.__fact[j]))

    self.__powr[0] = 1.
    for j in range(1, self.N_on + 1):
      self.__powr[j] = self.__powr[j-1] * (1. + self.bkg_ratio)
      if debug:
         print('__power[%d] = (1. + %.6f)^%d = %.6f' %\
             (j, self.bkg_ratio, j, self.__powr[j]))

    for j in range(self.N_on + 1):
      self.__bincoeff[j] = sp.comb(self.N_off+j, j)
      if debug:
         print('__bincoeff[%d] = bincoeff(%d, %d) = %.6f' %\
             (j, self.N_off+j, j, self.__bincoeff[j]))

    temp = np.float128(0.)
    for j in range(self.N_on + 1):
      term = self.__powr[self.N_on-j] * self.__bincoeff[j]
      if debug:
         print('__power[%d] * __bincoeff[%d] = %.6f' %\
             (self.N_on-j, j, term))
      temp += term

    self.__norm = 1. / temp
    if debug:
       print('__norm = 1 / %.6f = %.6e' %\
           (temp, self.__norm))

  def __ComputeCoefficientsW(self):
    """Compute the polynomial coefficients of the Likelihood derivative"""
    if self.__W[self.N_on] != 0.:
      return
    # computing directly the W coefficients would be less efficient
    self.__ComputeCoefficientsV()
    for j in range(self.N_on + 1):
      self.__W[j] = (j + 1.) * self.__V[j+1] - self.__V[j]

  def __ComputeCoefficientsV(self):
    """Compute the polynomial coefficients of the Likelihood function"""
    if self.__V[self.N_on] != 0.:
      return
    if self.__norm == 0.:
      self.__PrepareCache()
    for j in range(self.N_on + 1):
      num_j = self.__powr[j] * self.__bincoeff[self.N_on - j]
      self.__V[j] = self.__norm * num_j / self.__fact[j]
      if debug:
         print('__V[%d] = __norm * __powr[%d] * __bincoeff[%d] / __fact[%d] = %.6f' %\
             (j, j, self.N_on-j, j, self.__V[j]))

  def __ComputeCoefficientsU(self):
    """Compute the polynomial coefficients of the Likelihood integral"""
    if self.__U[self.N_on] != 0.:
      return
    # computing directly the U coefficients would be less efficient
    self.__ComputeCoefficientsV()
    for j in range(self.N_on, -1, -1):
      self.__U[j] = (j + 1) * self.__U[j+1] - self.__V[j]

  def __Get(self, x=0.):
    """Compute the value of the Likelihood function at x"""
    if (x < 0.):
      return 0.
    if self.N_on == 0:
      return np.exp(-x)
    self.__ComputeCoefficientsV()

    return np.exp(-x) * Polynomial(self.__V, x)

  def __GetSlope(self, x=0.):
    """Compute the slope of the Likelihood function at x"""
    if (x < 0.):
      return 0.
    if self.N_on == 0:
      return -np.exp(-x)
    self.__ComputeCoefficientsW()

    return np.exp(-x) * Polynomial(self.__W, x)

  def __GetIntegral(self, x=0.):
    """Compute the integral of the Likelihood function between 0 and x"""
    if (x <= 0.):
      return 0.
    if self.N_on == 0:
      return 1. - np.exp(-x)
    self.__ComputeCoefficientsU()

    return np.exp(-x) * Polynomial(self.__U, x) - self.__U[0]

  def __Maximize(self, accuracy=1.e-3):
    """Find the peak of the Likelihood function to some accuracy"""
    """ """
    """Since the value and location of the peak are not returned,"""
    """but stored within the object, the only reason for calling"""
    """externally this method is to anticipate this maximization"""
    # set a minimum and maximum accuracy
    if accuracy > .1:
      accuracy = .1
    elif accuracy < best_accuracy:
      accuracy = best_accuracy
    # this includes the special case N_on=0
    if self.bkg_ratio * self.N_on <= self.N_off:
      self.__accuracy = 0. # this result is exact
      self.__arg_max = 0.
    else:
      # numerically find the optimum using bisections
      self.__accuracy = accuracy
      lo = 0.
      # first guess is the asymptotic value for large N's
      hi = self.N_on - (self.N_off / self.bkg_ratio)
      #print 'BLike.__Maximize:slope(hi)=%.2f lo=%.2f hi=%.2f' % (self.__GetSlope(hi), lo, hi)
      while self.__GetSlope(hi) > 0.:
        #print 'BLike.__Maximize:slope(hi)=%.2f lo=%.2f hi=%.2f' % (self.__GetSlope(hi), lo, hi)
        lo = hi
        hi *= 1.5
      while hi - lo > self.__accuracy:
        mid = (hi + lo) / 2.
        if self.__GetSlope(mid) > 0.:
          lo = mid
        else:
          hi = mid
        #print 'BLike.__Maximize:slope(mid)=%.2f mid=%.2f lo=%.2f hi=%.2f' % (self.__GetSlope(mid), mid, lo, hi)
      self.__arg_max = (hi + lo) / 2.
    self.__max_like = self.__Get(self.__arg_max)

  def __GetBackground(self, mu_src):
    """Compute the value of the most likely expected counts from background"""
    """in the On measure, for a given expected counts from the source."""
    R = np.sqrt(mu_src**2 + 2 * self.__Am * mu_src + self.__Ap**2)
    # these maximise the combined likelihood function, accounting also for
    # the outcome of the Off measure and the bkg ratio
    return R / 2. + self.__Ap / 2. - mu_src / 2.

  def __GetLogProfile(self, mu_src):
    """Compute the logarithm of the profile Likelihood function for a given"""
    """expected counts from the source alone"""
    mu_bkg = self.__GetBackground(mu_src)
    mu_tot = mu_src + mu_bkg * (self.bkg_ratio + 1.)
    # the profile likelihood is normalised at 1 in its maximum
    return (self.N_on + self.N_off - mu_tot) + \
        self.N_off * np.log(mu_bkg / self.__base_off) + \
        self.N_on * np.log((mu_src + mu_bkg) / self.__base_on)

  def Significance(self, return_Pvalue=False):
    """Compute the significance of the signal [Zhang & Ramsden (1989)]"""
    # the Null Hypothesis is no-signal.
    # we see N_on counts expecting mu_on and N_off expecting bkg_ratio*mu_on
    # overall we see N_on/(N_on+N_off) counts expecting 1/(1+bkg_ratio)
    if self.N_on == 0:
      Pval = 1.
    else:
      N_tot = self.N_on + self.N_off
      Pval = st.binom.sf(self.N_on-1, N_tot, 1./(1.+self.bkg_ratio))

    if return_Pvalue:
       return Pval
    # convert the P-value into a Z-score (two-sided)
    return PvalToZscore(Pval)

  def MostLikely(self, accuracy=1.e-3, marginal=True):
    """Return the value that maximizes the Likelihood"""
    # this is the exact maximum of the profile likelihood
    if not marginal:
      if self.N_off > self.bkg_ratio * self.N_on:
        return 0.
      else:
        return self.N_on - self.N_off / self.bkg_ratio

    # set a minimum and maximum accuracy
    if accuracy > .1:
      accuracy = .1
    elif accuracy < best_accuracy:
      accuracy = best_accuracy
    # use cache if possible
    if accuracy < self.__accuracy or self.__max_like < 0.:
      self.__Maximize(accuracy)

    return self.__arg_max

  def MaxLike(self, accuracy=1.e-3):
    """Return the value of the Likelihood function at the peak"""
    # set a minimum and maximum accuracy
    if accuracy > .1:
      accuracy = .1
    elif accuracy < best_accuracy:
      accuracy = best_accuracy
    # use cache if possible
    if accuracy < self.__accuracy or self.__max_like < 0.:
      self.__Maximize(accuracy)

    return self.__max_like

  def Integrate(self, x_min=0., x_max=0.):
    """Return the probability that the expected counts are in the requested range"""
    if (x_max < x_min):
      return -1 * self.Integrate(x_max, x_min)
    if (x_max <= 0):
      return 0.
    if (x_min < 0):
      x_min = 0.

    return self.__GetIntegral(x_max) - self.__GetIntegral(x_min)

  def PDF(self, x=0.):
    """Compute the posterior probability distribution function at x"""

    if np.isinf(x):
      return 0.
    return self.__Get(x)

  def CDF(self, x=0.):
    """Compute the posterior cumulative distribution function at x"""

    if np.isinf(x):
      return 1.
    return self.__GetIntegral(x)

  def SF(self, x=0.):
    """Compute the posterior survival function at x"""

    if np.isinf(x):
      return 0.
    return 1. - self.__GetIntegral(x)

  def Quantile(self, coverage=.954499736104, accuracy=1e-3):
    """Compute the quantile function with the specified accuracy"""
    # special cases
    if coverage <= 0:
      return 0.
    if coverage >= 1:
      return float("inf")
    # set a minimum and maximum accuracy
    if accuracy > .1:
      accuracy = .1
    elif accuracy < best_accuracy:
      accuracy = best_accuracy

    lo = 0.
    hi = 1.
    while self.__GetIntegral(hi) < coverage:
      lo = hi
      hi *= 1.5
    while hi - lo > accuracy:
      mid = (hi + lo) / 2.
      if self.__GetIntegral(mid) > coverage:
        hi = mid
      else:
        lo = mid
    return (hi + lo) / 2.

  def ISF(self, coverage=.954499736104, accuracy=1e-3):
    """Compute the inverse survival function with the specified accuracy"""

    return self.Quantile(1. - coverage, accuracy)

  def TS(self, x=0.):
    """Compute the value of the Test Statistic (TS = 2*logLike) at x"""
    """TS<=0 and TS=0 only at the peak of the profile likelihood"""
    if (x < 0.):
      return -1. * float("inf")

    return 2. * self.__GetLogProfile(x)

  def ErrorBar(self, up=True, coverage=.682689492137, accuracy=1e-3):
    """Compute the uncertainty in the maximum Likelihood estimator"""
    """as a sided error bar, with a given accuracy and coverage"""
    """ """
    """The uncertainty is always positive, as long as it can be"""
    """computed with the requested coverage."""
    """If the peak is too close to zero, this method returns -1."""
    """In this case, consider computing an upper limit instead"""
    # set a minimum and maximum accuracy
    if accuracy > .1:
      accuracy = .1
    elif accuracy < best_accuracy:
      accuracy = best_accuracy
    # set a minimum and maximum coverage
    if coverage < min_coverage:
      coverage = min_coverage
    elif coverage > max_coverage:
      coverage = max_coverage
    # use cache if possible
    if accuracy < self.__accuracy or self.__max_like < 0.:
      self.__Maximize(accuracy)
    # a tricky way to manage both upper and lower bars at once
    sign = -1
    if up:
      sign = 1
    # each error bar has half of the total coverage
    I_exp = self.__GetIntegral(self.__arg_max) + sign * coverage / 2.
    # if the peak is too close to 0 for this coverage
    if I_exp < 0.:
      return -1.
    lo = 0.
    hi = np.sqrt(self.N_on + (self.N_off / (self.bkg_ratio**2)))
    while sign * self.__GetIntegral(self.__arg_max + sign * hi) < sign * I_exp:
      lo = hi
      hi *= 1.5
    while hi - lo > accuracy:
      mid = (hi + lo) / 2.
      if sign * self.__GetIntegral(self.__arg_max + sign * mid) < sign * I_exp:
        lo = mid
      else:
        hi = mid
    return (hi + lo) / 2.

  def UpperLimit(self, coverage=.954499736104, accuracy=1e-3):
    """Compute an upper limit with the specified coverage and accuracy"""
    # set a minimum and maximum coverage
    if coverage < min_coverage:
      coverage = min_coverage
    elif coverage > max_coverage:
      coverage = max_coverage

    return self.Quantile(coverage, accuracy)

  def HDI(self, coverage=.682689492137, accuracy=1e-5):
    """Compute the highest density interval with the specified coverage and accuracy"""
    # set a minimum and maximum accuracy
    if accuracy > .1:
      accuracy = .1
    elif accuracy < best_accuracy:
      accuracy = best_accuracy
    # set a minimum and maximum coverage
    if coverage < min_coverage:
      coverage = min_coverage
    elif coverage > max_coverage:
      coverage = max_coverage

    if self.CDF(self.MostLikely()) < coverage:
      lo = 0.
      hi = self.Quantile(coverage, accuracy)
      if self.__GetSlope(0.) < 0:
        return lo, hi
    else:
      lo = self.__arg_max - self.ErrorBar(False, coverage, accuracy)
      hi = self.__arg_max + self.ErrorBar(True, coverage, accuracy)

    # relative difference in density level
    delta = (self.PDF(hi) - self.PDF(lo)) / self.__max_like
    # shift size
    dx = (hi - lo) / 3.
    while np.abs(delta) > accuracy:
      if delta > 0.: # shift the interval up
        lo += dx
        if lo > self.__arg_max:
          lo = self.__arg_max
      else:
        lo -= dx
      hi = self.Quantile(self.CDF(lo) + coverage, accuracy)
      delta = (self.PDF(hi) - self.PDF(lo)) / self.__max_like
      # gradually shrink the shift
      dx *= .75

    return lo, hi
