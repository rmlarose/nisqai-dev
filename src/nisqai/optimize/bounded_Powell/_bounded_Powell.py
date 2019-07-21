from numpy import pi, copy, array
from scipy.optimize import minimize_scalar, OptimizeResult


def gramschmidt(basis):
    """
    basis: list of vectors
    returns: orthonormal list of vectors with first vector unchanged.
    """
    def proj(j, v):
        return j * (j.transpose() @ v)
    f = []
    for i in range(len(basis)):
        u = basis[i].copy()
        u0 = u.copy()
        for j in f:
            u -= proj(j, u0)
        f.append(array(normalize(u)))
    return f


def m_min(a, b):
    if a is None: return b
    elif b is None: return a
    else: return min(a, b)
def m_max(a, b):
    if a is None: return b
    elif b is None: return a
    else: return max(a, b)


def line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    x0 is the vector representing the current location
    alpha is the unit vector representing the direction
    
    ie the direction is along the line from x0 to alpha

    lower_bound is a list/array of the lower bounds for each parameter in x0.
    upper_bound is a list/array of the upper bounds for each parameter in x0.
    
    returns (lmin, lmax), the bounds for 
            lower_bound[i] <= x0_i+alpha_i*l <= upper_bound[i] 
        for all i.
    """
    # figure out how far forward we can go before we are out of bounds
    # for one of the params
    
    lmin, lmax = None, None
    for i in range(len(alpha)):
        if alpha[i] > 0:
            lmin = m_max(lmin, (lower_bound[i]-x0[i])/alpha[i])
            lmax = m_min(lmax, (upper_bound[i]-x0[i])/alpha[i])
        elif alpha[i] < 0:
            lmin = m_max(lmin, (upper_bound[i]-x0[i])/alpha[i])
            lmax = m_min(lmax, (lower_bound[i]-x0[i])/alpha[i])
    
    if lmin is None: lmin = 0
    if lmax is None: lmax = 0
    return lmin, lmax


class Result:
    def __init__(self):
        self.fevals, self.nfev, self.xs = [], 0, []
        self.iters, self.nit = [], 0
        self.directions, self.direc = [], None
        self.x, self.fun, self.message = [], 0, ""

    def add_x(self, x):
        self.xs.append(x)
        
    def add_f(self, f):
        self.fevals.append(f)
        self.nfev += 1
        
    def add_iter(self, i):
        self.iters.append(i)
        self.nit += 1
        
    def add_direction(self, direction):
        self.directions.append(direction)
        
    def get_f(self, back=0):
        return self.fevals[-1-back]
    
    def get_x(self, back=0):
        return self.xs[-1-back]
    
    def done(self, message=""):
        self.fun = min(self.fevals)
        i = self.fevals.index(self.fun)
        self.x = self.xs[i]
        self.message = message if message else self.message
        self.nit = len(self.iters)
        self.direc = self.directions[-1]
        
    def add(self, other):
        self.fevals.extend(other.fevals)
        self.nfev += other.nfev
        self.xs.extend(other.xs)
        self.iters.extend(other.iters)
        self.directions.extend(other.directions)
        self.message += other.message
        
    def __str__(self):
        s = "fun: %g\n" % self.fun
        s += "message: %s\n" % self.message
        s += "nfev: %d\n" % self.nfev
        s += "nit: %d\n" % self.nit
        s += "x: %s\n" % self.x
        s += "direc: %s" % self.direc
        return s
        
        
def normalize(vec):
    m = sum(x**2 for x in vec)**0.5
    return [x / m for x in vec]


def _bounded_Powell(fun, x0, args, direc, ftol, xtol, maxls,
                    maxfev, callback, lower_bound, upper_bound, gs):
    
    directions = [list(d) for d in direc]
    
    f = lambda x: fun(x, *args)

    res = Result()
    res.add_x(x0)
    res.add_f(f(x0))
    res.add_iter(res.get_f())
    
    options = dict(xatol=xtol)
    if maxls is not None: options["maxiter"] = maxls
        
    while True:
        res.add_direction(directions)
            
        directions_weight = [0.0]*len(directions)
        alpha_iter = [0.0]*len(x0)
        x00 = x0.copy()
        f00 = res.get_f()
        for j in range(len(directions)):
            if res.nfev > maxfev:
                res.done("Max function evaluations exceeded")
                return res            
            d = directions[j]
            bounds = line_for_search(x00, d, lower_bound, upper_bound)
            x = lambda l: [x00[i]+l*d[i] for i in range(len(x00))]
            new_f = lambda l: f(x(l))
            r = minimize_scalar(new_f, method="bounded", bounds=bounds, 
                                options=options)
            res.nfev += r.nfev - 1 # minus 1 because the next line adds one
            res.add_f(r.fun)
            res.add_x(x(r.x))
            for i in range(len(x0)): alpha_iter[i] += (res.xs[-1][i] - x0[i])
            directions_weight[j] = abs(r.x)
            x00 = res.get_x()
            
        if callback: callback(copy(x00))
            
        res.add_iter(res.get_f())
            
        if abs(res.get_f() - f00) / max(abs(f00),abs(res.get_f()), 1/2) < ftol:    
            res.done("Optimization converged")
            return res
        elif max(
            abs(x00[i]-x0[i]) for i in range(len(x0))
        ) / max(max(abs(x) for x in x0), max(abs(x) for x in x00), 1/2) < xtol:
            res.done("Optimization converged")
            return res
        elif res.nfev > maxfev:
            res.done("Optimization did not converge")
            return res
        
        j = directions_weight.index(max(directions_weight))
        directions.pop(j)
        directions.append(normalize(alpha_iter))
        
        # gram scmidt
        if gs:
            directions = list(reversed(directions))
            directions = gramschmidt(directions)
            directions = [list(d) for d in directions]
            directions.reverse()
        
        x0 = res.get_x().copy()
    
    return res


def _local_bounded_Powell(fun, x0, args, direc, ftol, xtol, maxls, 
                          maxfev, callback, lower_bound, upper_bound, gs):
    directions = [list(d) for d in direc]
    
    f = lambda x: fun(x, *args)

    res = Result()
    res.add_x(x0)
    res.add_f(f(x0))
    res.add_iter(res.get_f())
    
    options = dict(xatol=xtol)
    if maxls is not None: options["maxiter"] = maxls
        
    while True:
        l_bound = [max(x0[i]-pi/3, lower_bound[i]) for i in range(len(x0))]
        u_bound = [min(x0[i]+pi/3, upper_bound[i]) for i in range(len(x0))]
        
        res.add_direction(directions)
            
        directions_weight = [0.0]*len(directions)
        alpha_iter = [0.0]*len(x0)
        x00 = x0.copy()
        f00 = res.get_f()
        for j in range(len(directions)):
            if res.nfev > maxfev:
                res.done("Max function evaluations exceeded")
                return res            
            d = directions[j]
            bounds = line_for_search(x00, d, l_bound, u_bound)
            x = lambda l: [x00[i]+l*d[i] for i in range(len(x00))]
            new_f = lambda l: f(x(l))
            r = minimize_scalar(new_f, method="bounded", bounds=bounds, 
                                options=options)
            res.nfev += r.nfev - 1
            res.add_f(r.fun)
            res.add_x(x(r.x))
            for i in range(len(x0)): alpha_iter[i] += (res.xs[-1][i] - x0[i])
            directions_weight[j] = abs(r.x)
            x00 = res.get_x()
            
        if callback: callback(copy(x0))
            
        res.add_iter(res.get_f())
            
        if abs(res.get_f() - f00) / max(abs(f00),abs(res.get_f()), 1/2) < ftol:    
            res.done("Optimization converged")
            return res
        elif sum(
            (x00[i]-x0[i])**2 for i in range(len(x0))
        )**.5 / sum(a**2 for a in x0)**.5 < xtol:
            res.done("Optimization converged")
            return res
        elif res.nfev > maxfev:
            res.done("Optimization did not converge")
            return res
        
        j = directions_weight.index(max(directions_weight))
        directions.pop(j)
        directions.append(normalize(alpha_iter))
        
        # gram scmidt
        if gs:
            directions = list(reversed(directions))
            directions = gramschmidt(directions)
            directions = [list(d) for d in directions]
            directions.reverse()
        
        x0 = res.get_x().copy()
    
    return res



def bounded_Powell(f, x0, args=(), **kwargs):
    """
    Find the global minimium of a f with a bounded_Powell method.
        - Joseph Iosue

    f: function of x and args to minimized. Called with f(x, *args)
    x0: list type. Initial guess of parameters.
    args : tuple, optional. Extra arguments passed to the objective function.
    maxfev : int, maximum number of function evaluations to perform.
                  Note that the qPowell methods may go slightly over this
                  limit, since they will not halt in the middle of an 
                  iteration. The default is 2000.
    xtol: Relative tolerance for the parameters for convergence. The default is
          1e-4 (same as SciPy's Powell method).
    ftol: Relative tolerance for the function for convergence. The default is
          1e-4 (same as SciPy's Powell method).
    direc: initial set of direction vectors for the qPowell methods. The
           default are the standard basis unit vectors along each 
           parameter's directions.
    maxls: int, maximum function evaluations used at each linesearch. Default
                is no max.
    method: str, which qPowell method to use. The default is None or "", which 
                 defaults to just the standard qPowell method. The other option
                 is "locally-bounded".
    lower_bound: list or array of numbers. lower_bound[i] is the lower bound
                                           of the i^th parameter in x0. If no
                                           lower_bound is provided, then -pi
                                           will be used.
    upper_bound: list or array of numbers. upper_bound[i] is the upper bound
                                           of the i^th parameter in x0. If no
                                           upper_bound is provided, then pi
                                           will be used.
    callback: function of x, called after each iteration.
    gs: boolean, whether or not to use gram schmidt to keep direcs orthogonal.
        
    Returns scipy.minimize.OptimizeResult
        See scipy's doucmentation for a description of attributes.

    Examples usage:

    >>> from scipy.optimize import minimize
    >>> from bounded_Powell import bounded_Powell
    >>> res = minimize(
    >>>     [function],
    >>>     [initial params],
    >>>     options={
                'method'='locally_bounded', 
                'maxfev': 500,
                'xtol': 1e-2,
                'ftol': 1e-2,
                ...
            },
    >>>     method=bounded_Powell,
    >>>     callback=[callback function]
    >>> )
    """
    
    kwargs = kwargs.copy()
    
    callback = kwargs.pop("callback", None)
    method = kwargs.pop("method", "")
    maxfev = kwargs.pop("maxfev", 2000)
    xtol = kwargs.pop("xtol", 1e-4)
    ftol = kwargs.pop("ftol", 1e-4)
    direc = kwargs.pop(
        "direc", 
        [[0.0]*i + [1.0] + [0.0]*(len(x0)-i-1) for i in range(len(x0))]
    )  
    maxls = kwargs.pop("maxls", None)
    lower_bound = kwargs.pop("lower_bound", [-pi]*len(x0))
    upper_bound = kwargs.pop("upper_bound", [pi]*len(x0))
    gs = kwargs.pop("gs", False)

        
    if method == "locally-bounded":
        res = _local_bounded_Powell(f, x0, args, direc, ftol, xtol, maxls, 
                                    maxfev, callback, lower_bound, upper_bound, gs)
        
    elif not method or method == "bounded_Powell":
        res = _bounded_Powell(f, x0, args, direc, ftol, xtol, maxls, 
                              maxfev, callback, lower_bound, upper_bound, gs)
        
    else:
        raise ValueError(
            "bounded_Powell `method` must be either None, 'qPowell', "
            "or 'locally-bounded'."
        )
    

    return OptimizeResult(
        fun=res.fun,
        x=res.x,
        nit=res.nit, 
        nfev=res.nfev,
        message=res.message,
        direc=array(res.direc)
    )
    
    
def test_bounded_Powell():
    
    from numpy import sin, cos, random
    from scipy.optimize import minimize
    
    fun = lambda p: (
        (cos(p[0]*p[2]) + sin(p[1]) + (cos(p[2]) + sin(p[3]))**2 + 
        (cos(p[4])**2 + sin(p[5]*p[4]))**3)**5 + cos(p[6]**2)**3
    )
    
    x0 = random.random(7)
    
    maxfev = 2000
    assert minimize(fun, x0, method=bounded_Powell,
                    options=dict(maxfev=maxfev, gs=False)).fun < -240

    assert minimize(fun, x0, method=bounded_Powell,
                    options=dict(maxfev=maxfev, gs=True)).fun < -240

    assert minimize(fun, x0, method=bounded_Powell,
                    options=dict(maxfev=maxfev, gs=False, 
                                 method="locally-bounded")).fun < -240

    assert minimize(fun, x0, method=bounded_Powell,
                    options=dict(maxfev=maxfev, gs=True, 
                                 method="locally-bounded")).fun < -240
