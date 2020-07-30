"""
Check a univariate random number generator, graphically and via chi**2.

2020-07-30 (Tom Loredo):  Pulled out from a BB1Gamma test script
"""

from numpy import *
from matplotlib.pyplot import *


def check_rng(sampler, pdf, n_samp, n_bins, n_in=5, log_log=False):
    """
    CHeck a pseudo-random number generator `sampler` by generating samples
    and comparing their histogram with predictions from the density function
    `pdf`, both graphically and via chi-squared.

    To predict the counts in the `n_bins` bins, trapezoid rule quadrature is
    used with `n_in` nodes within each bin (so `n_in` + 2 nodes are used in
    total, per bin).
    """
    # TODO:  Refactor common code!
    samps = array([sampler() for i in range(n_samp)])
    # Linear axis case, best for signed samples.
    if not log_log:
        cts, bins, patches = hist(samps, bins=n_bins, log=False, alpha=.7)
        l, u = bins[0], bins[-1]
        xvals = linspace(l, u, n_bins+1 + n_bins*n_in)
        # Plot predicted number in bin subintervals.
        pdfs = pdf(xvals)
        diffs = diff(xvals)
        # *** should shift xvals .5 sub-bin size
        plot(xvals[:-1], n_samp*pdfs[:-1]*diffs*(n_in+1), 'k-', lw=2)
        # Plot predicted number in bins.
        pdxns = empty(n_bins)
        pvars = empty_like(pdxns)
        pdfs = n_samp*pdfs  # now norm = n_samp
        chi = 0.
        for i in range(n_bins):
            j = i*(n_in + 1)
            k = j + n_in + 1
            vals = 0.5*(pdfs[j:k] + pdfs[j+1:k+1])*diffs[j:k]
            pdxns[i] = vals.sum()
            pvars[i] = pdxns[i]  # Poisson; should be close to binomial here
            chi += (cts[i] - pdxns[i])**2/pvars[i]
        centers = bins[:-1] + .5*diff(bins)
        # plot(centers, pdxns, 'ro', mew=0)
        errorbar(centers, pdxns, sqrt(pdxns), fmt='ro', mew=0)
        l, u = ylim()
        ylim(0, u)
        s = r'$\chi^2_{%i} = %f$' % (n_bins-1, chi)
        text(.1, .8, s, fontsize=14, transform=gca().transAxes)

    # Log-log axes, for positive-valued samples.
    else:
        mask = samps > 0.
        lsamps = log10(samps[mask])
        l, u = samps[mask].min(), samps[mask].max()
        lbins = logspace(log10(l), log10(u), n_bins+1)
        cts, bins, patches = hist(samps[mask], bins=lbins, log=True, alpha=.7)
        centers = bins[:-1] + .5*diff(bins)
        bdr = centers[1]/centers[0]  # bin dyn range
        gca().set_xscale("log")
        xvals = logspace(log10(l), log10(u), n_bins+1 + n_bins*n_in)
        # Plot predicted number in bin subintervals.
        pdfs = pdf(xvals)
        diffs = diff(xvals)
        # For log-spaced bins (const bdr), the predicted counts in a bin at x,
        # using the trapezoid rule, is ~ p(x) * 2*x*(bdr-1.)/(bdr+1).
        loglog(xvals, 2.*n_samp*xvals*pdfs*(bdr-1.)/(bdr+1.), 'k-', lw=2)
        pdxns = empty(n_bins)
        pvars = empty_like(pdxns)
        pdfs = n_samp*pdfs  # now norm = n_samp
        chi = 0.
        for i in range(n_bins):
            j = i*(n_in + 1)
            k = j + n_in + 1
            vals = 0.5*(pdfs[j:k] + pdfs[j+1:k+1])*diffs[j:k]
            pdxns[i] = vals.sum()
            pvars[i] = pdxns[i]  # Poisson; should be close to binomial here
            chi += (cts[i] - pdxns[i])**2/pvars[i]
        # plot(centers, pdxns, 'ro', mew=0)
        errorbar(centers, pdxns, sqrt(pdxns), fmt='oC3', mew=0)
        yl, yu = ylim()
        ylim(.5*pdxns.min(), 1.5*pdxns.max())
        s = r'$\chi^2_{%i} = %4.1f$' % (n_bins-1, chi)
        text(.1, .8, s, fontsize=14, transform=gca().transAxes)
