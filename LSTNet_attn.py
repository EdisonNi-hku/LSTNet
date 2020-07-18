import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTNet_attn(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.convChannel = args.convChannel
        self.total_l = args.total_l
        self.arI = args.arI
        self.m = data.m
        self.kernel_l = args.kernel_l
        self.skip = args.skip
        self.pt = int((self.total_l - self.kernel_l) / self.skip)
        self.rnnH = args.rnnH
        self.attn = args.attn
        self.conv1 = nn.Conv2d(1, self.convChannel, kernel_size=(self.kernel_l, self.m))
        self.gru1 = nn.GRU(self.convChannel, self.rnnH)
        self.dropout = nn.Dropout(p=args.dropout)
        # self.gruSkip = nn.GRU(self.convChannel, self.rnnS)
        self.linear1 = nn.Linear((self.attn + 1) * self.rnnH, self.m)
        self.ar = nn.Linear(self.arI, 1)

    def forward(self, x):
        # x (batch_size, total_time, width)
        batch_size = x.size(0)

        # CNN (batch_size, channel, total_time, width)
        c = x.view(-1, 1, self.total_l, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        # 去除width
        c = torch.squeeze(c, 3)

        # RNN (time_step, batch_size, CNN_output_channel)
        r = c.permute(2, 0, 1).contiguous()
        out, r = self.gru1(r)
        # 去除num_of_layers
        r = self.dropout(torch.squeeze(r, 0))

        '''
        # skip-rnn
        s = c[:, :, int(-self.pt * self.skip):].contiguous()
        # skip每一行代表一个skip_period内的数据，pt代表每个period
        s = s.view(batch_size, self.convChannel, self.pt, self.skip)
        s = s.permute(2, 0, 3, 1).contiguous()
        # pt维度保证了间隔
        s = s.view(self.pt, batch_size * self.skip, self.convChannel)
        _, s = self.gruSkip(s)
        s = s.view(batch_size, self.skip * self.rnnS)
        s = self.dropout(s)
        r = torch.cat((r, s), 1)
        '''
        # Attention
        out = out.permute(1, 0, 2).contiguous()
        out = out[:, -2 - self.attn:-2, :]
        last = out[:, -2:-1, :]
        out = out.permute(0, 2, 1).contiguous()
        weight = torch.bmm(last, out)
        weight = weight.permute(0, 2, 1).contiguous()
        out = out.permute(0, 2, 1).contiguous()
        out = out * weight
        # out (batch_size, attention + 1, rnnH)
        out = torch.cat((out, last), 1)
        out = out.view(batch_size, (self.attn + 1) * self.rnnH)

        res = self.linear1(out)

        # AR
        # AR模型中纳入回归的时间序列
        z = x[:, -self.arI:, :]
        z = z.permute(0, 2, 1).contiguous().view(-1, self.arI)
        z = self.ar(z)
        z = z.view(-1, self.m)
        res = res + z

        # (batch_size, width)
        return res