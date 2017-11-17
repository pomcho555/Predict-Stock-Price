import numpy as np 
from network import Network
from get_data import Get_data


data_code = "EIA/PET_RWTC_D"  # Quandl data code

n_prev = 50                   # how much data to use to predict
hidden_size = 300             # how big hidden layer size
learning_epochs = 300         # learning epochs



if __name__ == "__main__":
    model = Network(n_prev, hidden_size)
    data_from = Get_data(n_prev, data_code)
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.08)

    raw_data = data_from.get_raw_data()

    for learn in range(learning_epochs):
        for day in range(n_prev, len(raw_data)-1):
            model.zero_grad()
            model.hidden = model.init_hidden()

            learn_material, target = data_from.get_data(day)
            
            model_predict = model(learn_material)

            loss = loss_func(model_predict, target)
            loss.backward()
            optimizer.step()



    

