import numpy as np
from itertools import *
from functools import *

def iterate(f, x):
    yield x
    yield from iterate(f, f(x))

def until_convergence(initial):
    return accumulate(initial, no_repeat)

def no_repeat(previous, current):
    if (previous == current).all():
        raise StopIteration
    else:
        return current

def until_nearly_convergent(initial, tolerance=0.001):
    return accumulate(
        initial,
        partial(within_tolerance, tolerance)
    )

def within_tolerance(tolerance, previous, current):
    if abs(previous - current) < tolerance:
        raise StopIteration
    else:
        return current

def new_means(points, old_means):
    k = old_means.shape[0]
    assignments = closest_index(points, old_means)
    new_thing = np.array([
        cluster_mean(points[assignments == cluster_index])
        for cluster_index in range(k)
    ])
    return new_thing

def closest_index(points, means, p=2):
    distances = np.array([
        np.linalg.norm(points - mean, p, axis=1)
        for mean in means
    ])
    return np.stack(distances, axis=-1).argmin(axis=1)
        
def cluster_mean(points):
    return points.sum(axis=0)/points.shape[1]
        
def k_means(points, k):
    initial_means = points[[
        np.random.randint(0, points.shape[1]) for _ in range(k)
    ], :]
    return iterate(
        partial(new_means, points),
        initial_means
    )
