import torch
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim 
import torch.autograd as autograd 


class Network(nn.Module):
    def __init__(self, n_prev, hidden_size):
        super(Network, self).__init__()
        self.n_prev = n_prev
        self.hidden_size = hidden_size

        self.LSTM_layer = nn.LSTM(self.n_prev, self.hidden_size)
        self.hidden = self.init_hidden()
        self.hidden_to_sub = nn.Linear(self.hidden_size, 50)
        self.sub_to_predict = nn.Linear(50, 1)

    
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))
        
    def forward(self, data):
        out, hidden = self.LSTM_layer(data)
        out = self.hidden_to_sub(out)
        prediction = self.sub_to_predict(out)
        return prediction


# check
print(Network(70, 300))