import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

df = pd.read_csv('data_stocks.csv')
df = df.loc[:10316, :]
df = df.drop(columns=['DATE'])
print(df.shape)


class Data_util(object):
    def __init__(self, train_size, valid_size, window, horizon, cuda):
        self.cuda = cuda
        self.data = df.to_numpy()
        self.window = window
        self.horizon = horizon
        self.n, self.m = self.data.shape
        self._normalized()
        self._split(int(train_size * self.n), int((train_size + valid_size) * self.n))

    def _normalized(self):
        for i in range(self.m):
            self.data[:, i] = self.data[:, i] / np.max(np.abs(self.data[:, i]))

    def _split(self, train, valid):
        train_set = range(self.window + self.horizon - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.window, self.m))
        Y = torch.zeros((n, self.m))

        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.window
            X[i, :, :] = torch.from_numpy(self.data[start:end, :])
            Y[i, :] = torch.from_numpy(self.data[idx_set[i], :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.tensor(range(length))
            index = index.long()
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if self.cuda:
                X = X.cuda()
                Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size