"""
At the moment, this module only contains the minimize function, which
is mostly equivalent to the scipy.optimize.minimize function, except that
we add the "bounded_Powell" minimization method option. To see how to use
it, run help(nisqai.optimize.bounded_Powell).
"""

from ._minimize import minimize

__all__ = "minimize",
__str__ = "optimize"
