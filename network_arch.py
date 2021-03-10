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
        short_term , out1 , out2 = self.LSTMnet(x_batch)
        loss1 = self.criterion1(out1, y2_batch)
        loss2 = self.criterion2(out2, y1_batch)
        loss= loss1 + loss2
        loss.backward()
        self.opt.step()
        return short_term , out1 , out2, loss1.detach() , loss2.detach()

    def val_step(self, args, x_batch, y1_batch, y2_batch  ):

        self.opt.zero_grad()
        with torch.no_grad():
            short_term , out1 , out2 = self.LSTMnet(x_batch)
            loss1 = self.criterion1(out1, y2_batch)
            loss2 = self.criterion2(out2, y1_batch)
            loss= loss1 + loss2

        return short_term , out1 , out2, loss1.detach() , loss2.detach()

    def test_step(self, args, x_batch, y1_batch, y2_batch  ):

        short_term , out1 , out2 = self.LSTMnet(x_batch)
        loss1 = self.criterion1(out1, y2_batch)
        loss2 = self.criterion2(out2, y1_batch)
        loss= loss1 + loss2

        return short_term , out1 , out2, loss1.detach() , loss2.detach()

    def implement(self, args, x_batch):
        short_term , out1 , out2 = self.LSTMnet(x_batch)
        return short_term , out1 , out2


class LSTMClassifier(nn.Module):

    def __init__(self, args):
        super(LSTMClassifier , self).__init__()
        self.input_dim = args.input_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.lstm_layer_dim = args.lstm_layer_dim
        self.layer1_class = args.layer1_class
        self.layer2_class = args.layer2_class
        self.layer3_class = args.layer3_class
        self.layer1_reg = args.layer1_reg
        self.seq_dim = args.seq_dim

        #Define the network structure
        #LSTM layer
        self.rnn = nn.LSTM(self.input_dim, self.lstm_hidden_dim, self.lstm_layer_dim, dropout=0.2, batch_first=True)
        #Classification
        self.fc1 = nn.Linear(self.lstm_hidden_dim*self.seq_dim, self.layer1_class)
        self.fc2 = nn.Linear(self.layer1_class, self.layer2_class)
        self.fc3 = nn.Linear(self.layer2_class, self.layer3_class)
        #Translation
        self.fc4 = nn.Linear(self.lstm_hidden_dim*self.seq_dim, self.layer1_reg)
        self.fc5 = nn.Linear(self.layer1_reg, self.seq_dim)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        #LSTM
        h0, c0 = self.init_hidden(x)
        for i in range(x.size(1)):
            x_c = x[:,i:i+1, :]
            out, (hn, cn) = self.rnn(x_c, (h0, c0))
            if i==0:
                out_total= out
            else:
                out_total = torch.cat([out_total,out],1)
        out = out_total.reshape(-1, self.lstm_hidden_dim * self.seq_dim )

        #Classification
        output_class1 = F.relu(self.fc1(out))
        out_drop_class1 = self.dropout(output_class1)
        output_class2 = F.relu(self.fc2(out_drop_class1))
        out_drop_class2 = self.dropout(output_class2)
        output_class3 = self.fc3(out_drop_class2)

        #Translation
        output_reg1 = F.relu(self.fc4(out))
        output_reg2 = self.fc5(output_reg1)
        output_reg3 = output_reg2.reshape(-1, self.seq_dim, 1)

        return out_total, output_class3 , output_reg3

    def init_hidden(self, x):
        h0 = torch.zeros(self.lstm_layer_dim, x.size(0), self.lstm_hidden_dim)
        c0 = torch.zeros(self.lstm_layer_dim, x.size(0), self.lstm_hidden_dim)
        return [t.cuda() for t in (h0, c0)]
