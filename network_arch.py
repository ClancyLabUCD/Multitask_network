import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Network():

    def __init__(self, args):
        self.lr = args.lr
        self.LSTMnet = LSTMClassifier(args)
        self.LSTMnet.cuda()
        #Optimzer
        self.opt = optim.Adam(self.LSTMnet.parameters(), lr=self.lr)
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.MSELoss()

    def train_step(self, args, x_batch, y1_batch, y2_batch  ):

        self.opt.zero_grad()
        out1 , out2 = self.LSTMnet(x_batch)
        loss1 = self.criterion1(out1, y2_batch)
        loss2 = self.criterion2(out2, y1_batch)
        loss= loss1 + loss2
        loss.backward()
        self.opt.step()
        return loss1.detach() , loss2.detach()

    def test_step(self, args, x_batch, y1_batch, y2_batch  ):

        out1 , out2 = self.LSTMnet(x_batch)
        loss1 = self.criterion1(out1, y2_batch)
        loss2 = self.criterion2(out2, y1_batch)
        loss= loss1 + loss2

        return out1 , out2, loss1.detach() , loss2.detach()


class LSTMClassifier(nn.Module):

    def __init__(self, args):
        super(LSTMClassifier , self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.layer_dim = args.layer_dim
        self.output_dim1 = args.output_dim1
        self.output_dim2 = args.output_dim2
        self.output_dim3 = args.output_dim3
        self.output_dim4 = args.output_dim4
        self.seq_dim = args.seq_dim

        #Define the network structure
        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim*self.seq_dim, self.output_dim1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(self.output_dim1, self.output_dim2)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc3 = nn.Linear(self.output_dim2, self.output_dim3)
        self.fc4 = nn.Linear(self.hidden_dim*self.seq_dim, self.output_dim4)
        self.fc5 = nn.Linear(self.output_dim4, self.seq_dim)

    def forward(self, x):
        #LSTM
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = out.reshape(-1, self.hidden_dim * self.seq_dim )

        #Classification
        output11 = F.relu(self.fc1(out))
        out_drop1 = self.dropout1(output11)
        output12 = F.relu(self.fc2(out_drop1))
        out_drop2 = self.dropout2(output12)
        out1 = self.fc3(out_drop2)

        #Regression
        out12 = F.relu(self.fc4(out))
        out22 = self.fc5(out12)
        out2 = out22.reshape(-1, self.seq_dim, 1)

        return out1 , out2

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]
