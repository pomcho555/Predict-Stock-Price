import quandl



# Get the data dynamics as below
#mydata_ = quandl.get("EIA/PET_RWTC_D")
#plt.xlabel("date")
#plt.ylabel("Price")
#plt.plot(mydata_)
#plt.show()

class Get_data:
    def __init__(self, n_prev, data_code):
        self.n_prev = n_prev
        self.data = quandl.get(data_code, returns="numpy")
    
    def get_data(self, today):
        X = []
        for day in range(self.n_prev):
            X.append(self.data[today-self.n_prev+day][1])
        target = self.data[today][1]
        return X, target

    def get_raw_data(self):
        return self.data
