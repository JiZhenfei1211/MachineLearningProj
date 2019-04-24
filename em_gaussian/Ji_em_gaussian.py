#!/usr/bin/env python3
import numpy as np
import math

if not __file__.endswith('_em_gaussian.py'):
    print(
        'ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)

# DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)
DATA_PATH = './data/'
LIKELIHOOD_THRESHOLD = 1e-3


def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9 * len(data))
    train_xs = np.asarray(data[:dev_cutoff], dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:], dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs


def init_mus(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9 * len(data))
    train_xs = np.asarray(data[:dev_cutoff], dtype=dtype)
    cov = np.cov(train_xs.T)
    mean = np.sum(train_xs, axis=0, keepdims=True) / train_xs.shape[0]
    mean = mean.T
    mus = np.append(mean, mean, axis=1)
    for i in range(args.cluster_num - 2):
        mus = np.append(mus, mean, axis=1)
    return mus.T, cov


def init_model(args):
    if args.cluster_num:
        lambdas = np.zeros(args.cluster_num) + 1 / args.cluster_num
        # mus, cov = init_mus(args)
        mus = np.random.rand(args.cluster_num, 2)
        sigmas = []
        if not args.tied:
            for i in range(args.cluster_num):
                cov = np.random.rand(2, 2) * 1.5
                sigmas.append(np.dot(cov, cov.T))
            sigmas = np.asarray(sigmas)
        else:
            cov = np.random.rand(2, 2)
            sigmas.append(cov)
            sigmas = np.asarray(sigmas).reshape((2, 2))
    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file, 'r') as f:
            for line in f:
                # each line is a cluster, and looks like this:
                # lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float, line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    # TODO: do whatever you want to pack the lambdas, mus, and sigmas into the model variable (just a tuple, or a class, etc.)
    # NOTE: if args.tied was provided, sigmas will have a different shape
    model = (lambdas, mus, sigmas)
    return model


def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
    K = args.cluster_num
    N = train_xs.shape[0]
    r = np.zeros((K, N))
    lambdas, mus, sigmas = model
    for it in range(args.iterations):
        # E step
        s = np.zeros(N)
        for i in range(N):
            temp = np.zeros(K)
            xi = train_xs[i]
            for k in range(K):
                if not args.tied:
                    temp[k] = lambdas[k] * multivariate_normal(mean=mus[k], cov=sigmas[k]).pdf(xi)
                else:
                    temp[k] = lambdas[k] * multivariate_normal(mean=mus[k], cov=sigmas).pdf(xi)
                s[i] += temp[k]
            for k in range(K):
                r[k][i] = temp[k] / s[i]

        # M step
        for k in range(K):
            # calculate lamdas[k]
            lambdas[k] = np.sum(r[k]) / N

            # calculate mus[k]
            total = np.zeros(mus.shape[1])
            for i in range(N):
                total += r[k][i] * train_xs[i]
            mus[k] = total / np.sum(r[k])

            # calculate sigmas[k]
            summ = np.zeros((train_xs.shape[1], train_xs.shape[1]))
            for i in range(N):
                if train_xs[i].ndim == 1:
                    data_temp = train_xs[i].reshape(train_xs.shape[1], 1)
                    mu_temp = mus[k].reshape(mus.shape[1], 1)
                    diff_temp = data_temp - mu_temp
                    summ += r[k][i] * np.dot(diff_temp, diff_temp.T)
                else:
                    summ += r[k][i] * np.dot(train_xs[i] - mus[i], (train_xs[i] - mus[i]).T)
            if not args.tied:
                sigmas[k] = summ / np.sum(r[k])
            else:
                sigmas = summ / np.sum(r[k])

        if not args.nodev:
            new_ll_dev = average_log_likelihood(model, dev_xs, args)
            if new_ll_dev - ll_dev <= LIKELIHOOD_THRESHOLD:
                break
            else:
                ll_dev = new_ll_dev

    model = (lambdas, mus, sigmas)
    return model


def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    # TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    ll = 0.0
    lambdas, mus, sigmas = model
    K = args.cluster_num
    N = data.shape[0]
    for i in range(N):
        temp = 0
        xi = data[i]
        for k in range(K):
            if not args.tied:
                temp += lambdas[k] * multivariate_normal(mean=mus[k], cov=sigmas[k]).pdf(xi)
            else:
                temp += lambdas[k] * multivariate_normal(mean=mus[k], cov=sigmas).pdf(xi)
        ll += log(temp)
    return ll / N


def extract_parameters(model):
    # TODO: extract lambdas, mus, and sigmas from the model and return them (same type and shape as in init_model)
    lambdas, mus, sigmas = model
    return lambdas, mus, sigmas


def main():
    import argparse
    import os
    print('Gaussian')  # Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true',
                        help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied', action='store_true',
                        help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print(
            'You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str, a))

        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '), mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '), map(lambda s: np.nditer(s), sigmas)))))


if __name__ == '__main__':
    main()
