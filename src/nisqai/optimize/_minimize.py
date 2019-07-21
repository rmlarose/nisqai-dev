#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


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
