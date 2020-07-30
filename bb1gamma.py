"""
A generalization of the gamma function targeting the Schecter function regime
(corresponding to the gamma distribution's shape parameter alpha < 1). In this
regime, the gamma function is a falling power law with an exponential cutoff at
large values of the argument. This "break by one" generalization includes a
factor that smoothly adjusts the power law index at small arguments so that the
distribution remains normalized for alpha > -1, which includes the [-1, 0]
regime used to model galaxy luminosity function and stellar initial mass
functions.

Created 2014-07-14 by Tom Loredo; adapted from powerlaws.py
"""

from numpy import *
from numpy import random
from scipy import stats
import scipy.special as sp


__all__ = ['BB1TruncPL']


class BB1Gamma:
    """
    'Break-By-1 Gamma distribution'

    A distribution and RNG for a non-negative quantity, x, with (approximate)
    falling power law behavior between lower and upper scales, and smooth
    truncation beyond those scales.  At large values of x, the pdf decays
    exponentially.  At small values, the power law becomes shallower (by a
    factor x, i.e., with index decreased by 1).

    The power law slope, Gamma(x) = d(log p)/d(log x) = (x/p) dp/dx, obeys

        Gamma(x) = L(x) + beta - x/u,

    where L(x) -> 0 for x >> l, and L(x) -> 1 for x << l.  This corresponds
    to an exponential cutoff at large x, and a small x slope of beta+1.  It
    is proper for beta > -2.

    In detail, the pdf is proportional to

        (1 - exp(-x/l)) * x**beta * exp(-x/u),

    and the logarithmic slope is as specified above, with

        L(x) = (x/l) * exp(-x/l) / (1 - exp(-x/l)).

    If the L(x) factor and its assocated parameter, l, are omitted, this
    distribution becomes:

        * a gamma distribution with shape parameter alpha = beta + 1 and scale
          parameter u;

        * a Schecter function with power law index beta and cutoff scale u.

    The break-by-one gamma distribution is valid (i.e., proper) in the rising
    power law (beta > 0) regime, but closely resembles a gamma distribution
    when beta is large, so it is mainly useful in the -2 < beta < 0 regime.
    """

    # Euler-Mascheroni constant:
    gamma_const = - sp.digamma(1)

    # Constant from the small z expansion of the gamma function Gamma(z):
    k1 = (3*gamma_const**2 + 0.5*pi**2)/6.

    def __init__(self, beta, l, u):
        if beta > 0.:
            raise ValueError('Only beta <= 0 currently supported!')
        self.beta = beta
        self.beta1 = beta + 1.
        self.l, self.u = l, u
        self.r = u/l
        if beta > -1.001 and beta < -.999:
            self.norm = u * self._norm_approx()
        else:
            self.norm = u * sp.gamma(self.beta1) * (1 - 1./(1+self.r)**self.beta1)
        self.norm = 1. / self.norm

        # Values to support sampling:
        self.r_beta = self.r**beta
        self.gamma_dist = stats.gamma(beta+2., scale=u)
        if beta > -1.:  # split envelope only for this case
            c1 = (u/l) * sp.gamma(beta+2.) * sp.gammainc(beta+2., l/u)
            c2 = (l/u)**beta * exp(-l/u)
            self.p_gamma = c1/(c1+c2)
            self.p_exp = c2/(c1+c2)
            self.expon = stats.expon(scale=u)

    def _norm_approx(self):
        """
        Return the norm constant for BB1TruncPL as a function of power law index
        gamma and upper/lower ratio r, calculated with an approximation accurate
        to O(beta1**2) for beta1 = beta+1.

        For beta in [-1.001, -.999] and r = 1e4, this is good to a few
        parts in 1e7.  It is better for smaller r.
        """
        # r-dependent constants for the expansion in powers of beta1:
        C0 = log(self.r + 1.)
        C1 = -C0*(self.gamma_const + 0.5*C0)
        C2 = C0*(self.k1 + C0*(0.5*self.gamma_const + C0/6.))

        return C0 + self.beta1*(C1 + self.beta1*C2)

    def pdf(self, x):
        """
        Return the PDF at x.
        """
        y = x/self.u
        return self.norm * (1.-exp(-x/self.l)) * y**self.beta * exp(-y)

    def log_slope(self, x):
        """
        Return the logarithmic slope of the PDF at x, i.e., 

            Gamma(x) = d(log p)/d(log x) = (x/p) dp/dx,

        where p(x) is the PDF.  There are three regimes, lower (< l),
        intermediate (l < x < u), and upper (x > u).  Specifically,

            Gamma(x) = L(x) + beta - x/u

        with

            L(x) = (x/l) * exp(-x/l) / (1 - exp(-x/l)).
        """
        z = x/self.l
        emz = exp(-z)
        return z*emz/(1. - emz) + self.beta - x/self.u

    def sample(self, return_n=False):
        """
        Return a single sample from the distribution.  If return_n=True, also
        return the number of proposals needed to generate the sample; this is
        useful for estimating the sampling efficiency.

        This is a modification of the gamma sampler algorithm of Ahrens and 
        Dieter (1974).
        """
        ntry = 0
        # For gamma <= -1, just use a gamma envelope for rejection.
        if self.beta <= -1.:
            while True:
                ntry += 1
                x = self.gamma_dist.rvs()
                r = x/self.l
                u = random.rand()
                if u < (1. - exp(-r))/r:
                    break
            if return_n:
                return x, ntry
            else:
                return x

        # For -1 < gamma < 0, split the envelope into gamma and exponential parts.
        u = random.rand()
        if u < self.p_gamma:  # gamma distn envelope
            while True:
                ntry += 1
                x = self.gamma_dist.rvs()
                r = x/self.l
                u = random.rand()
                if u < (1. - exp(-r))/r:
                    break
            if return_n:
                return x, ntry
            else:
                return x
        else:  # exponential distn envelope
            while True:
                ntry += 1
                x = self.expon.rvs()
                u = random.rand()
                if u < (1. - exp(-x/self.l)) * (x/self.u)**self.beta * self.r_beta:
                    break
            if return_n:
                return x, ntry
            else:
                return x
