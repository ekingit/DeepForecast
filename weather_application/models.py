import torch
import torch.nn as nn


class local_LSTM(nn.Module):
    def __init__(self, num_inputs, num_hiddens,num_layers=1,num_out=1):
        super().__init__()
        self.lstm = nn.LSTM(num_inputs, num_hiddens, num_layers, batch_first=True)
        self.linear = nn.Linear(num_hiddens, num_out)
    def forward(self, X, H, c, target_seq_len):
        pred_list = []
        state, (H,c) = self.lstm(X,(H,c)) #TODO: state is not used. reformulation for computational efficiency?
        pred = self.linear(H[0]) #prediction for the next day
        Z = X.clone()
        pred_list.append(pred)
        for j in range(1,target_seq_len): #prediction for the (j+1)th day
          Z = torch.cat([Z[:,1:],pred.unsqueeze(-1)],1) #concatinate last target_seq with the pred
          state, (H,c) = self.lstm(Z,(H,c)) # Checked! state[:,-1,:] = H[0]
          pred = self.linear(H[0])
          pred_list.append(pred)
        pred_tens = torch.stack(pred_list,1)
        return pred_tens # shape = (batch_size, target_seq_len, 1)
    

class RNN_periodic(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers=1, num_out=1):
        super().__init__()
        self.rnn = nn.RNN(num_inputs, num_hiddens, num_layers)
        self.linear = nn.Linear(num_hiddens, num_out)
    def forward(self, X, H):
        state, H = self.rnn(X,H)
        pred = self.linear(state)
        return pred, H
    
class LSTM_periodic(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers=1, num_out=1):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.lstm = nn.LSTM(num_inputs, num_hiddens, num_layers)
        self.linear = nn.Linear(num_hiddens, num_out)
    def forward(self, X, H,c):
        state, (H,c) = self.lstm(X, (H,c))
        pred = self.linear(state)
        return pred, (H, c)