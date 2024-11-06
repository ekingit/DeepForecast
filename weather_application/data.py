import pandas as pd
import torch


class Weather_Data():
    def __init__(self, source, column, train_years=8,val_years=1):
        self.df_raw = pd.read_csv(source)[column]
        self.normal = (self.df_raw-self.df_raw.mean())/self.df_raw.std()
        self.tens = torch.tensor(self.normal,dtype=torch.float).unsqueeze(-1)
        days_of_years = [0]+[sum(366 if i % 4 == 0 else 365 for i in range(year+1)) for year in range(0,10)]

        self.train = self.tens[0:days_of_years[train_years]]
        self.val = self.tens[days_of_years[train_years]:days_of_years[train_years+val_years]]
        self.test = self.tens[days_of_years[train_years+val_years]:]

    def data_chunks(self, N, k): #N = seq_len, k = target_seq_len (for k-step predictions)
        """ Creates input-output pairs from a sequence with a sliding window.
        Takes the first N elements as input (in_data) and the subsequent k elements as output (out_data), 
        then repeats this process by moving one element forward each time."""

        in_train = torch.stack([self.train[i:i+N] for i in range(len(self.train)-N-k+1)],0)
        out_train = torch.stack([self.train[i+N:i+N+k] for i in range(len(self.train)-N-k+1)])

        in_val = torch.stack([self.val[i:i+N] for i in range(len(self.val)-N-k+1)],0)
        out_val = torch.stack([self.val[i+N:i+N+k] for i in range(len(self.val)-N-k+1)],0)

        in_test = torch.stack([self.test[i:i+N] for i in range(len(self.test)-N-k+1)])
        out_test = torch.stack([self.test[i+N:i+N+k] for i in range(len(self.test)-N-k+1)])

        return (in_train, out_train), (in_val, out_val), (in_test, out_test)
    

class Sine_Data():
    def __init__(self, p, T, tau=0): #Create sinus wave of length T with period 2pi/p with phase pi*tau
        self.time = torch.arange(0, T, dtype=torch.float32)
        self.x = torch.sin((p* self.time+tau)*torch.pi)

    def data_chunks(self, N, k): #N=seq_len, k=target size (k-step predictions) k=0 
        in_data = torch.stack([self.x[i:i+N] for i in range(len(self.x)-N-k+1)],0)
        out_data = torch.stack([self.x[i+N:i+N+k] for i in range(len(self.x)-N-k+1)],0)
        return in_data, out_data 





    