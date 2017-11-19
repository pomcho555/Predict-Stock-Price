import numpy as np 
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.autograd as Variable
from network import Network
from get_data import Get_data


data_code = "EIA/PET_RWTC_D"  # Quandl data code

n_prev = 30                   # how much data to use to predict
hidden_size = 300             # how big hidden layer size
cell_size = 100               # how big cell layer size
learning_epochs = 300         # learning epochs


if __name__ == "__main__":
    model = Network(n_prev, hidden_size, cell_size)
    data_from = Get_data(n_prev, data_code)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    raw_data = data_from.get_raw_data()
    
    for learn in range(learning_epochs):
        for day in range(n_prev, len(raw_data) - 1):
            material, target = data_from.get_data(day)
            predict = model(material)
            loss = loss_function(predict, target)
            loss.backward()
            optimizer.step()
    
    
