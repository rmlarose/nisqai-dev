from scipy.optimize import minimize as scipy_minimize
from .bounded_Powell import bounded_Powell


def minimize(*args, **kwargs):
	"""
	TODO: make our own. For now, this is equivalent to the 
	scipy.optimize.minimize function. There is one exception;
	we have added a new minimizer method called "bounded_Powell".
	To see its options, see help(nisqi.optimize.bounded_Powell).
	"""

	kwargs = kwargs.copy()
	method = kwargs.pop("method", "COBYLA")
	if method == "bounded_Powell": method = bounded_Powell
	kwargs["method"] = method

	try: return scipy_minimize(*args, **kwargs)
	except ValueError as e: print(e)

	raise ValueError(
		"Method options are any of scipy.optimize.minimize method options "
		"or 'bounded_Powell'. To see all the optional arguments to 'bounded_Powell' "
		"see help(nisqi.optimize.bounded_Powell)."
	)
