import os
import sys
from joblib import Parallel, delayed
import argparse
import numpy as np
import joblib

def gen_data(n, d, base_fn, tau_fn, prop_fn, sigma, dist='uniform'):
    if dist == 'uniform':
        X = np.random.uniform(0, 1, size=(n, d))
        Xtest = np.random.uniform(0, 1, size=(10000, d))
    if dist == 'normal':
        X = np.random.normal(0, 1, size=(n, d))
        Xtest = np.random.normal(0, 1, size=(10000, d))
    if dist == 'centered_uniform':
        X = np.random.uniform(-.5, .5, size=(n, d))
        Xtest = np.random.uniform(-.5, .5, size=(10000, d))
    T = np.random.binomial(1, prop_fn(X))
    y = (T - .5) * tau_fn(X) + base_fn(X) + \
        sigma * np.random.normal(0, 1, size=(n,))
    return y, T, X, Xtest


def get_data_generator(setup='E', n=5000, d=10, sigma=0.5):
    if setup == 'A':
        dist = 'uniform'

        def base_fn(X):
            return np.sin(
                np.pi * X[:, 0] * X[:, 1]) + 2 * (X[:, 2] - .5) ** 2 + X[:, 3] + .5 * X[:, 4]

        def prop_fn(X):
            return np.clip(
                np.sin(np.pi * X[:, 0] * X[:, 1]), .2, .8)

        def tau_fn(X):
            return .2 + (X[:, 0] + X[:, 1]) / 2
    elif setup == 'B':
        dist = 'centered_uniform'

        def base_fn(X):
            return np.maximum(0, np.maximum(
                X[:, 0] + X[:, 1], X[:, 2])) + np.maximum(X[:, 3] + X[:, 4], 0)

        def prop_fn(X):
            return .5 * np.ones(X.shape[0])

        def tau_fn(X):
            return X[:, 0] + np.log(1 + np.exp(X[:, 1]))
    elif setup == 'C':
        dist = 'centered_uniform'

        def base_fn(X):
            return 2 * np.log(1 + np.exp(np.sum(X[:, :3], axis=1)))

        def prop_fn(X):
            return 1 / (1 + np.exp(X[:, 1] + X[:, 2]))

        def tau_fn(X):
            return np.ones(X.shape[0])
    elif setup == 'D':
        dist = 'centered_uniform'

        def base_fn(X):
            return .5 * (np.maximum(0, np.sum(X[:, :3], axis=1)) + np.maximum(0, X[:, 3] + X[:, 4]))

        def prop_fn(X):
            return 1 / (1 + np.exp(-X[:, 0]) + np.exp(-X[:, 1]))

        def tau_fn(X):
            return np.maximum(0, np.sum(X[:, :3], axis=1)) - np.maximum(0, X[:, 3] + X[:, 4])
    elif setup == 'E':
        dist = 'centered_uniform'

        def base_fn(X):
            return 5 * np.maximum(0, X[:, 0] + X[:, 1])

        def prop_fn(X):
            return 1 / (1 + np.exp(3 * X[:, 1] + 3 * X[:, 2]))

        def tau_fn(X):
            return 2 * ((X[:, 0] > 0.1) | (X[:, 1] > 0.1)) - 1
    elif setup == 'F':
        dist = 'centered_uniform'

        def base_fn(X):
            return 5 * np.maximum(0, X[:, 0] + X[:, 1])

        def prop_fn(X):
            return 1 / (1 + np.exp(3 * X[:, 1] + 3 * X[:, 2]))

        def tau_fn(X):
            return X[:, 0] + np.log(1 + np.exp(X[:, 1]))
    else:
        raise AttributeError(f"Invalid parameter setup={setup}")

    def gen_data_fn():
        return gen_data(
            n, d, base_fn, tau_fn, prop_fn, sigma, dist=dist)

    return gen_data_fn, base_fn, tau_fn, prop_fn


gen_data_fn, base_fn, tau_fn, prop_fn = get_data_generator()
samples= gen_data_fn()
x= samples[0]

samples= gen_data_fn()
y= samples[0]

print(x.shape, y.shape)
print(np.sum(x!=y))