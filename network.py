import torch
import torch.nn as nn 
import torch.autograd as autograd 
from torch.autograd import Variable
import numpy as np 


class Network(nn.Module):
    def __init__(self, n_prev, h0, c0):
        super(Network, self).__init__()
        self.n_prev = n_prev
        self.h0 = h0
        self.c0 = c0

        self.LSTM_layer = nn.LSTM(self.n_prev, self.h0, self.c0)
        self.fc1 = nn.Linear(self.h0, 300)
        self.fc2 = nn.Linear(300, 250)
        self.fc3 = nn.Linear(250, 150)
        self.fc4 = nn.Linear(150, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 20)
        self.fc7 = nn.Linear(20, 1)

    def forward(self, x):
        output, hn = self.LSTM_layer(x)
        out = self.fc1(output)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        predict = self.fc7(out)
        return predict

    def init_hidden(self):
        return 

#model = Network(1, 100, 50)
#a = np.array([1.5])
#a = Variable(torch.from_numpy(a)).float()
#a = a.view(1,1,1)
#print(model(a))