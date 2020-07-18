import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from data import Data_util
from LSTNet import LSTNet
from LSTNet_attn import LSTNet_attn
import torch.optim as optim

L1 = nn.L1Loss(reduction='sum')
L2 = nn.MSELoss(reduction='sum')


def train(model, optimizer, Data, X, Y, epoch, device, criterion, batch_size):
    model.train()
    batch_id = 0
    total_loss = 0
    n_samples = 0
    for data, target in Data.get_batches(X, Y, batch_size):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_samples += (output.size(0) * Data.m)
        if (batch_id + 1) % 30 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
        batch_id += 1
    return total_loss / n_samples


def evaluate(model, Data, X, Y, device, batch_size):
    model.eval()
    total_loss_l1 = 0
    total_loss_l2 = 0
    n_samples = 0
    predict = None
    test = None

    for data, target in Data.get_batches(X, Y, batch_size):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        if predict is None:
            predict = output
            test = target
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, target))

        total_loss_l1 += L1(output, target).item()
        total_loss_l2 += L2(output, target).item()
        n_samples += (output.size(0) * Data.m)

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)

    rse = math.sqrt(total_loss_l2) / math.sqrt(np.square(predict - mean_p).sum())

    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, correlation


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--modelType', type=int, default=1,
                    help='1 for LST-Skip, 0 for LST-Attn')
parser.add_argument('--convChannel', type=int, default=50,
                    help='number of CNN hidden units')
parser.add_argument('--rnnH', type=int, default=50,
                    help='number of RNN hidden units')
parser.add_argument('--total_l', type=int, default=391,
                    help='window size')
parser.add_argument('--kernel_l', type=int, default=6,
                    help='the kernel size of the CNN layers')
parser.add_argument('--arI', type=int, default=120,
                    help='The window size of the highway component')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model_skip.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=2)
parser.add_argument('--skip', type=float, default=77)
parser.add_argument('--attn', type=float, default=300)
parser.add_argument('--rnnS', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=False)
args = parser.parse_args()

Data = Data_util(0.6, 0.2,  args.total_l, args.horizon, args.cuda)

if args.L1Loss:
    criterion = nn.L1Loss(reduction='sum')
else:
    criterion = nn.MSELoss(reduction='sum')

if args.modelType == 1:
    model = LSTNet(args, Data)
else:
    model = LSTNet_attn(args, Data)

if args.cuda:
    model.cuda()
opt = optim.Adam(model.parameters())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
best_val = 10000000


for epoch in range(args.epochs):
    train_loss = train(model, opt, Data, Data.train[0], Data.train[1], epoch, device, criterion, args.batch_size)
    val_loss, val_corr = evaluate(model, Data, Data.valid[0], Data.valid[1], device, args.batch_size)
    print(
        '| end of epoch {:3d} | train_loss {:5.4f} | valid rse {:5.4f} | valid corr  {:5.4f}'.format(
            epoch, train_loss, val_loss, val_corr))
    # Save the model if the validation loss is the best we've seen so far.

    if val_loss < best_val:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val = val_loss
    if epoch % 5 == 0:
        test_acc, test_corr = evaluate(model, Data, Data.test[0], Data.test[1], device, args.batch_size)
        print("test rse {:5.4f} | test corr {:5.4f}".format(test_acc, test_corr))

with open(args.save, 'rb') as f:
    model = torch.load(f)
test_acc, test_corr = evaluate(model, Data, Data.test[0], Data.test[1], device, args.batch_size)
print("test rse {:5.4f} | test corr {:5.4f}".format(test_acc, test_corr))