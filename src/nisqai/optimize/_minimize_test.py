from nisqai.optimize import minimize
from numpy import sin, cos, random
    

def test_minimize():

    fun = lambda p: (
        (cos(p[0]*p[2]) + sin(p[1]) + (cos(p[2]) + sin(p[3]))**2 + 
        (cos(p[4])**2 + sin(p[5]*p[4]))**3)**5 + cos(p[6]**2)**3
    )
    
    x0 = random.random(7)
    
    # testing bounded_Powell
    maxfev = 2000
    assert minimize(fun, x0, method="bounded_Powell",
                    options=dict(maxfev=maxfev, gs=False)).fun < -240

    assert minimize(fun, x0, method="bounded_Powell",
                    options=dict(maxfev=maxfev, gs=True)).fun < -240

    assert minimize(fun, x0, method="bounded_Powell",
                    options=dict(maxfev=maxfev, gs=False, 
                                 method="locally-bounded")).fun < -240

    assert minimize(fun, x0, method="bounded_Powell",
                    options=dict(maxfev=maxfev, gs=True, 
                                 method="locally-bounded")).fun < -240

    # testing Powell
    assert minimize(fun, x0, method="Powell",
                    options=dict(maxfev=maxfev)).fun < -240
