"""
Demonstrate the break-by-one gamma distribution.

Created 2020-07-29 by Tom Loredo; based on bb1_plots.py
"""

from numpy import *
from scipy import stats
from scipy.special import gamma

from matplotlib.pyplot import *

from bb1gamma import BB1Gamma
from check_rng import check_rng

ion()

# From myplot.py:
rc('figure.subplot', bottom=.125, top=.95, right=.95)  # left=0.125
rc('font', size=14)  # default for labels (not axis labels)
rc('font', family='serif')  # default for labels (not axis labels)
rc('axes', labelsize=18)
rc('xtick.major', pad=8)
rc('xtick', labelsize=16)
rc('ytick.major', pad=8)
rc('ytick', labelsize=16)
rc('savefig', dpi=150)
rc('axes.formatter', limits=(-4,4))
# Use TeX labels with CMR font:
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})


class LRAxes:
    """
    Axes set for a plot with two ordinate axes (left and right) sharing a common
    abscissa axis.

    In matplotlib lingo, this is a two-scale plot using twinx().
    """

    def __init__(self, l=0.15, r=0.85, subplt=111, fig=None):
        """
        Define a LR axes set.

        If `fig` is not given, a new figure is created with size (8,6).

        If `fig` is a tuple, a new figure is created using `fig` as the size.

        Otherwise, `fig` is used as an existing figure instance for the plot.

        `subplt` specifies where in the figure to place the LR axes set, using
        the syntax of mpl's subplot() command.

        `l` and `r` specify locations of the left and right axes, in relative
        coordinates.  The default values are meant to allow separate labeling
        of the L and R ordinate axes.
        """
        if fig is None:  # new fig w/ default size
            self.fig = figure(figsize=(8,6))
        elif iterable(fig):  # new fig using figsize tuple
            self.fig = figure(figsize=fig)
        else:  # use passed fig
            self.fig = fig
        # Left and right axes:
        self.leftax = self.fig.add_subplot(subplt)
        self.fig.subplots_adjust(left=l, right=r)
        self.rightax = self.leftax.twinx()
        # Use thicker frame lines.
        self.leftax.patch.set_lw(1.25)  # thicker frame lines
        self.rightax.patch.set_lw(1.25)  # thicker frame lines
        # Leave with the left axes as current.
        self.fig.sca(self.leftax)

    def left(self):
        self.fig.sca(self.leftax)
        return self.leftax

    def right(self):
        self.fig.sca(self.rightax)
        return self.rightax


# A wide, 3-pane figure:
fig = figure(figsize=(18,5))
fig.tight_layout()
fig.subplots_adjust(wspace=.27)

# Left:  log p vs log x
ax_loglog = subplot(131)
# xlabel(r'$x$ $(\equiv L/u)$')
xlabel(r'$L$')
# ylabel('$p(L)$')
ylabel('$f(L)$')

# Middle:  x*log p vs log x
ax_slogx = LRAxes(l=0.05, r=.99, fig=fig, subplt=132)  # semilogx, x*p(x) vs. log(x)
ax_slogx.left()
xlabel(r'$L$')
# ylabel(r'$L \times p(L)$')
ylabel(r'$L \times f(L)$')
ax_slogx.right()
ylabel('Slope', labelpad=-5, rotation=-90)

alpha = .2  # gamma dist'n shape parameter; PL index is gamma = alpha - 1
scale = 100.
xlog = logspace(-3, 3, 300)
xlin = linspace(0.001, 300., 500)


# Some reference lines for beta = -.8:
# ax_loglog.loglog(xlog, (xlog/.1)**-.8, 'k--')
# ax_loglog.loglog(xlog, (xlog/.1)**.2, 'k--')


# First plot gamma dist'n, left & middle panes:
gd = stats.gamma(alpha, scale=scale)
pdf = gd.pdf(xlog)
ax_loglog.loglog(xlog, pdf, ':k', lw=2, label=r'$\alpha=0.2$')
ax_slogx.left()
semilogx(xlog, xlog*pdf, ':k', lw=2, label=r'$\alpha=0.2$')


# Helper for plotting BB1 instances in left, middle panes:

def plot_figs(bpl, xlog, c, lbl):

    pdf = bpl.pdf(xlog)
    slope = bpl.log_slope(xlog)

    ax_loglog.loglog(xlog, pdf, '-'+c, lw=2, label=lbl)

    ax_slogx.left()
    semilogx(xlog, xlog*pdf, '-'+c, lw=2, label=lbl)

    ax_slogx.right()
    semilogx(xlog, slope, '--'+c, lw=1)


# Middle power law range is x_l to scale.
x_l = .1


# BB1, beta = -.8 (valid range for gamma dist'n):
bb1 = BB1Gamma(-.8, x_l, scale)
lbl = r'$\beta=%3.1f$' % -.8
plot_figs(bb1, xlog, 'C0', lbl)

# BB1, beta = -1:
bb1 = BB1Gamma(-1., x_l, scale)
lbl = r'$\beta=%3.1f$' % -1
plot_figs(bb1, xlog, 'C1', lbl)

# BB1, beta = -1.2 (gamma dist'n would be improper):
bb1 = BB1Gamma(-1.2, x_l, scale)
lbl = r'$\beta=%3.1f$' % -1.2
plot_figs(bb1, xlog, 'C2', lbl)

# Legends and tics.
ax_loglog.legend(loc=0, fontsize='small')  # 0=best, 1=UR, 2=UL...
ax_slogx.left()
legend(loc='upper left', fontsize='small', framealpha=.8)
ax_slogx.right()
ylim(-2, 1)
yticks([-2, -1, 0, 1])


# 3rd pane for demonstrating the sampler.
subplot(133)

# Set up a BB1 instance:
scale = 100.
# beta = -0.8
beta = -1.2
bb1 = BB1Gamma(beta, .1, scale)
lbl = 'BB1 %3.1f' % beta

# For reproducibility, set RNG seed.
random.seed(42)

# Plot samples and check against PDF via chi**2:
check_rng(bb1.sample, bb1.pdf, 1000, 20, 10, True)
xlabel(r'$L$')
# ylabel(r'Counts, $C\times L\times p(L)$')
ylabel(r'Counts, $C\times L\times f(L)$')

# Some reference lines:
xlog = logspace(-3, 3, 300)
# loglog(xlog, 100.*(xlog/2.3)**(beta+1), 'k--')
# # loglog(xlog, 3.*(xlog/.01)**(beta+2), 'k--')  # beta = -.8
# loglog(xlog, 24.*(xlog/.008)**(beta+2), 'k--')  # beta = -1.2
