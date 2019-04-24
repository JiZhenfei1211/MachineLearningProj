#!/usr/bin/env python3
import numpy as np
from io import StringIO
import drawplot as plt
NUM_FEATURES = 124  # features are 1 through 123 (123 only in test set), +1 for the bias
# DATA_PATH = "/u/cs246/data/adult/" #TODO: if doing development somewhere other than the cycle server, change this to the directory where a7a.train, a7a.dev, and a7a.test are
DATA_PATH = "./data/"


# returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y, 0)  # treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature - 1] = value
    x[-1] = 1  # bias
    return y, x


# return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals], [v[1] for v in vals])
        return np.asarray([ys], dtype=np.float32).T, np.asarray(xs, dtype=np.float32).reshape(len(xs), NUM_FEATURES,
                                                                                              1)  # returns a tuple, first is an array of labels, second is an array of feature vectors


def init_model(args):
    w1 = None
    w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1, len(w2))
    else:
        # TODO (optional): If you want, you can experiment with a different random initialization. As-is, each weight is uniformly sampled from [-0.5,0.5).
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES)  # bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1)  # add bias column

    # At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.
    model = (w1, w2)
    return model


def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):
    X = train_xs.reshape(-1, NUM_FEATURES)
    Y = train_ys.reshape(-1, 1)
    m = X.shape[0]
    W_1 = model[0]
    W_2 = model[1]
    num_neurons = W_1.shape[0]
    loss = 0
    alpha = args.lr
    iterations = args.iterations
    A_1 = np.zeros((num_neurons + 1, 1))
    train_accuracy = []
    dev_accuracy = []
    for n in range(iterations):
        for i in range(m):
            x = X[i, :].reshape(-1, 1)
            y = Y[i, :].reshape(1, 1)
            Z_1 = np.dot(W_1, x)
            A_1[:-1, :] = sigmoid_function(Z_1)
            A_1[num_neurons, :] = 1
            Z_2 = np.dot(W_2, A_1)
            A_2 = sigmoid_function(Z_2)
            # loss = np.mean((y - A_2) ** 2)
            # print(loss)
            dZ_2 = (A_2 - y) * sigmoid_derivative(Z_2)
            dW_2 = np.dot(dZ_2, A_1.T)
            dZ_1 = np.multiply(np.dot(W_2[:, :-1].T, dZ_2), sigmoid_derivative(Z_1))
            dW_1 = np.dot(dZ_1, x.T)
            W_1 = W_1 - alpha * dW_1
            W_2 = W_2 - alpha * dW_2
        if not args.nodev:
            train_accuracy.append(test_accuracy((W_1, W_2), train_ys, train_xs))
            dev_accuracy.append(test_accuracy((W_1, W_2), dev_ys, dev_xs))
    # plt.draw_plot(iterations, alpha, num_neurons, train_accuracy, dev_accuracy)
    return W_1, W_2


def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid_function(z) * (1 - sigmoid_function(z))


def test_accuracy(model, test_ys, test_xs):
    num_neurons = model[0].shape[0]
    test_xs = test_xs.reshape(-1, NUM_FEATURES)
    total = test_ys.shape[0]
    A_1 = np.zeros((num_neurons + 1, total))
    W_1 = model[0]
    W_2 = model[1]
    Z_1 = np.dot(W_1, test_xs.T)
    A_1[:-1, :] = sigmoid_function(Z_1)
    A_1[num_neurons, :] = 1
    Z_2 = np.dot(W_2, A_1)
    A_2 = sigmoid_function(Z_2)
    A_2[A_2 > 0.5] = 1
    A_2[A_2 <= 0.5] = 0
    difference = test_ys - A_2.T
    total_correct = difference[difference == 0].shape[0]
    return total_correct / total


def extract_weights(model):
    W_1 = model[0].copy()
    W_2 = model[1].copy()
    return W_1, W_2


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1', 'W2'), type=str,
                               help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False,
                        help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH, 'a7a.train'),
                        help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH, 'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH, 'a7a.test'), help='Test data file.')

    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs = parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)

    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1, w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2, w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))


if __name__ == '__main__':
    main()
