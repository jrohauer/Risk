# Solving an integral using numerical methods
import numpy
from scipy.stats import norm
# Trapezoidal Rule - T
# Midpoint Rule - M
# Simpson’s Rule - S


a = -1000000
b = 0
n = 100000000


def f(x):
    fx = (1/numpy.sqrt(2*numpy.pi))*numpy.exp((-x**2)/2)

    return fx


def trapezoid(a, b, n):
    """
    trapezoidal rule numerical integral implementation
    a: interval start
    b: interval end
    n: number of steps
    return: numerical integral evaluation
    """

    # initialize result variable
    res = 0

    # calculate number of steps
    h = (b - a) / n

    # start at a
    x = a

    # sum 2*yi (1 ≤ i ≤ n-1)
    for _ in range(1, n):
        x += h
        res += f(x)
    res *= 2

    # evaluate function at a and b and add to final result
    res += f(a)
    res += f(b)

    # divide h by 2 and multiply by bracketed term
    return (h / 2) * res

def midpoint(a, b, n):
    """
    midpoint rule numerical integral implementation
    a: interval start
    b: interval end
    n: number of steps
    return: numerical integral evaluation
    """

    # initialize result variable
    res = 0

    # calculate number of steps
    h = (b - a) / n

    # starting midpoint
    x = a + (h / 2)

    # evaluate f(x) at subsequent midpoints
    for _ in range(n):
        res += f(x)
        x += h

    # multiply final result by step size
    return h * res


def simpsons(a, b, n):
    """
    simpson's rule numerical integral implementation
    a: interval start
    b: interval end
    n: number of steps
    return: numerical integral evaluation
    """

    # calculate number of steps
    h = (b - a) / n

    # start at a
    x = a

    # store f(x) evaluation at start and end
    res = f(x)
    res += f(b)

    # sum [2*yi (i is even), 4*yi (i is odd)]
    for i in range(1, n):
        x += h
        res += 2 * f(x) if i % 2 == 0 else 4 * f(x)

    # divide h by 3 and multiply for final result
    return (h / 3) * res