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
    

class ResBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_size)
        self.lin1 = nn.Linear(input_size, input_size // 2)
        self.norm2 = nn.LayerNorm(input_size // 2)
        self.lin2 = nn.Linear(input_size // 2, output_size)
        self.lin3 = nn.Linear(input_size, output_size)
        self.act = nn.ELU()

    def forward(self, X):
        X = self.act(self.norm1(X))
        res = self.lin3(X)
        X = self.act(self.norm2(self.lin1(X)))
        X = self.lin2(X)
        return X + res
    
    
class regularized_local_LSTM(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size):
        super().__init__() 
        self.input_mlp = nn.Sequential(  # (input_size, seq_len) 
            nn.Linear(seq_len, 2 * seq_len), # --> (input_size,4*seq_len) 
            nn.ELU(),
            nn.Linear(2 * seq_len, 64)) # --> (input_size,128)
        #LSTM
        self.lstm = nn.LSTM(64, hidden_size, num_layers=1)
        #resblocks
        blocks = ResBlock(64, 64)
        self.blocks = nn.Sequential(blocks)
        #output
        self.out = nn.Linear(64, output_size)
        self.act = nn.ELU()
    def forward(self, X, H_in, c_in): #data(batch_size, seq_len) --> (batch_size,1)
        X = X.squeeze(-1) #(batch_size, seq_len,1) --> (batch_size, seq_len)
        X_vect = self.input_mlp(X).unsqueeze(0) #(batch_size, seq_len) --> (1,batch_size,128)
        Y_hat, (H_out, c_out) = self.lstm(X_vect, (H_in, c_in)) #Y_hat.shape(1,batch_size,128)
        X = self.act(self.blocks(Y_hat)).squeeze(0) #(1,batch_size,128) --> (batch_size,128)
        X = self.out(X) # (batch_size,128) --> (batch_size,1)
        return X, (H_out, c_out)
