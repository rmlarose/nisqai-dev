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
                    options=dict(maxfev=maxfev, gs=False)).fun < -200

    assert minimize(fun, x0, method="bounded_Powell",
                    options=dict(maxfev=maxfev, gs=True)).fun < -200

    assert minimize(fun, x0, method="bounded_Powell",
                    options=dict(maxfev=maxfev, gs=False, 
                                 method="locally-bounded")).fun < -200

    assert minimize(fun, x0, method="bounded_Powell",
                    options=dict(maxfev=maxfev, gs=True, 
                                 method="locally-bounded")).fun < -200

    # testing Powell
    assert minimize(fun, x0, method="Powell",
                    options=dict(maxfev=maxfev)).fun < -200
