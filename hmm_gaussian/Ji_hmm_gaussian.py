#!/usr/bin/env python3
import numpy as np

if not __file__.endswith('_hmm_gaussian.py'):
    print(
        'ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

# DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)
DATA_PATH = "./data/"


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


def init_model(args):
    if args.cluster_num:
        mus = np.random.rand(args.cluster_num, 2)
        if not args.tied:
            sigmas = [np.eye(2) for k in range(args.cluster_num)]
        else:
            sigmas = np.eye(2)
        sigmas = np.asarray(sigmas)
        transitions = np.ones((args.cluster_num,
                               args.cluster_num))  # transitions[i][j] = probability of moving from cluster i to cluster j
        transitions /= np.sum(transitions, axis=1)
        initials = np.ones(args.cluster_num)  # probability for starting in each state
        initials /= np.sum(initials)
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file, 'r') as f:
            for line in f:
                # each line is a cluster, and looks like this:
                # initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float, line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5], vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    model = (initials, mus, sigmas, transitions)
    return model


def forward(model, data, args):
    from math import log
    initials, mus, sigmas, transitions = model

    log_likelihood = 0.0
    T = data.shape[0]
    N = args.cluster_num
    alphas = np.zeros((N, T))

    for i in range(N):
        if args.tied:
            alphas[i, 0] = initials[i] * normPDF(data[0, :], mus[i, :], sigmas)
        else:
            alphas[i, 0] = initials[i] * normPDF(data[0, :], mus[i, :], sigmas[i])
    log_likelihood += log(np.sum(alphas[:, 0]))
    alphas[:, 0] /= np.sum(alphas[:, 0])

    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                if args.tied:
                    alphas[i, t] += alphas[j, t - 1] * transitions[j, i] * normPDF(data[t, :], mus[i, :], sigmas)
                else:
                    alphas[i, t] += alphas[j, t - 1] * transitions[j, i] * normPDF(data[t, :], mus[i, :], sigmas[i])
        alphas_sum = np.sum(alphas[:, t])
        alphas[:, t] /= alphas_sum
        log_likelihood += log(alphas_sum)

    return alphas, log_likelihood


def backward(model, data, args):
    initials, mus, sigmas, transitions = model
    T = data.shape[0]
    N = args.cluster_num
    betas = np.zeros((N, T))

    for i in range(N):
        betas[i, T - 1] = 1;
    betas[:, T - 1] /= np.sum(betas[:, T - 1])

    for t in range(T - 2, -1, -1):
        for i in range(N):
            for j in range(N):
                if args.tied:
                    betas[i, t] += betas[j, t + 1] * transitions[i, j] * normPDF(data[t + 1, :], mus[j, :], sigmas)
                else:
                    betas[i, t] += betas[j, t + 1] * transitions[i, j] * normPDF(data[t + 1, :], mus[j, :], sigmas[j])
        betas[:, t] /= np.sum(betas[:, t])

    return betas


def normPDF(x, mu, sigma):
    from scipy.stats import multivariate_normal
    return multivariate_normal(mean=mu, cov=sigma).pdf(x)


def train_model(model, train_xs, dev_xs, args):
    if not args.nodev:
        _, dev_ll = average_log_likelihood(model, dev_xs, args)

    initials, mus, sigmas, transitions = model
    T = train_xs.shape[0]
    N = args.cluster_num

    xi = np.zeros((N, N, T))
    for it in range(args.iterations):
        alphas, train_ll = forward(model, train_xs, args)
        betas = backward(model, train_xs, args)
        gamma = np.zeros((N, T))
        for k in range(N):
            gamma[k, :] = alphas[k, :] * betas[k, :] / np.sum((alphas * betas), axis=0)

        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    if args.tied:
                        xi[i, j, t] = alphas[i, t] * transitions[i, j] * normPDF(train_xs[t + 1], mus[j], sigmas) * \
                                      betas[j, t + 1]
                    else:
                        xi[i, j, t] = alphas[i, t] * transitions[i, j] * normPDF(train_xs[t + 1], mus[j], sigmas[j]) * \
                                      betas[j, t + 1]
            xi[:, :, t] /= np.sum(xi[:, :, t])

        initials = gamma[:, 0]
        tempsum = np.sum(xi, axis=2)
        transitions = tempsum / np.sum(gamma, axis=1)[None].T
        if args.tied:
            sigmas = np.zeros((2, 2))
        else:
            sigmas = np.zeros((N, 2, 2))
        gamma_sum = np.sum(gamma, axis=1)
        for n in range(N):
            mus[n, :] = np.sum(gamma[n, :] * train_xs.T, axis=1) / gamma_sum[n]
            x_mu = (train_xs - mus[n, :][None]).T
            if args.tied:
                sigmas += np.dot(x_mu, (x_mu * gamma[n, :]).T) / gamma_sum[n]
            else:
                sigmas[n] = np.dot(x_mu, (x_mu * gamma[n, :]).T) / gamma_sum[n]

        model = (initials, mus, sigmas, transitions)
        if not args.nodev:
            _, new_dev_ll = average_log_likelihood(model, dev_xs, args)
            if abs(new_dev_ll - dev_ll) < 0.005:
                break;

    return model


def average_log_likelihood(model, data, args):
    _, log_likelihood = forward(model, data, args)
    return log_likelihood / len(data)


def extract_parameters(model):
    initials, mus, sigmas, transitions = model
    return initials, transitions, mus, sigmas


def main():
    import argparse
    import os
    print('Gaussian')  # Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
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
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str, a))

        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '), transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '), mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '), map(lambda s: np.nditer(s), sigmas)))))


if __name__ == '__main__':
    main()
