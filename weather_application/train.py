import torch

def train_lstm(model, dl, optimizer, loss, hidden_size, num_layers, device = 'cpu'):
    target_seq_len = dl.dataset[0][1].shape[0]
    train_loss = 0
    model = model.to(device)
    model.train()
    for X, Y in dl:
        optimizer.zero_grad()
        X, Y = X.to(device), Y.to(device)
        H = torch.zeros(num_layers,X.shape[0],hidden_size,device=device)
        c = torch.zeros(num_layers,X.shape[0],hidden_size,device=device)
        Y_hat = model(X,H,c,target_seq_len)
        batch_loss = loss(Y_hat,Y) #loss for backpropagation
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()
    train_loss = train_loss/(len(dl.dataset)*target_seq_len)
    return train_loss


def test_lstm(model, dl, loss, hidden_size, num_layers, device='cpu'):
    target_seq_len = dl.dataset[0][1].shape[0] 
    test_loss = 0
    model = model.to(device)
    model.eval()
    for X, Y in dl:
        X, Y = X.to(device), Y.to(device)
        H = torch.zeros(num_layers,X.shape[0], hidden_size,device=device)
        c = torch.zeros(num_layers,X.shape[0], hidden_size,device=device)
        Y_hat = model(X,H,c,target_seq_len)
        batch_loss = loss(Y_hat, Y)
        test_loss += batch_loss.item() #mean of number of examples
    test_loss = test_loss/(len(dl.dataset)*target_seq_len) # mean of total
    return test_loss 



def train_test_RNN(model, X, Y, split_year, optimizer, criterion, num_layers, hidden_size, device='cuda'):
    model.train()
    optimizer.zero_grad()
    X, Y = X.to(device), Y.to(device)
    model = model.to(device)
    H = torch.zeros(num_layers,hidden_size,device=device)
    days_of_years = [0]+[sum(366 if i % 4 == 0 else 365 for i in range(year+1)) for year in range(0,10)]
    
    Y_hat, H = model(X[0:days_of_years[split_year]], H) 
    train_loss = criterion(Y[0:days_of_years[split_year]],Y_hat)
    train_loss.backward()
    optimizer.step()
    train_loss = train_loss.item()
    model.eval()
    Y_hat, H = model(X[days_of_years[split_year]:days_of_years[split_year+1]],H)
    val_loss = criterion(Y[days_of_years[split_year]:days_of_years[split_year+1]],Y_hat).item()

    Y_hat, H = model(X[days_of_years[split_year+1]:],H)
    test_loss = criterion(Y[days_of_years[split_year+1]:],Y_hat).item()
    return train_loss, val_loss, test_loss


def train_test_LSTM(model, X, Y, split_year, optimizer, criterion, num_layers, hidden_size, device='cuda'):
    model.train()
    optimizer.zero_grad()
    X, Y = X.to(device), Y.to(device)
    model = model.to(device)
    H = torch.zeros(num_layers,hidden_size,device=device)
    c = torch.zeros(num_layers,hidden_size,device=device)
    days_of_years = [0]+[sum(366 if i % 4 == 0 else 365 for i in range(year+1)) for year in range(0,10)]
    
    Y_hat, (H, c) = model(X[0:days_of_years[split_year]], H, c) 
    train_loss = criterion(Y[0:days_of_years[split_year]],Y_hat)
    train_loss.backward()
    optimizer.step()
    train_loss = train_loss.item()
    model.eval()
    Y_hat, (H, c) = model(X[days_of_years[split_year]:days_of_years[split_year+1]],H,c)
    val_loss = criterion(Y[days_of_years[split_year]:days_of_years[split_year+1]],Y_hat).item()

    Y_hat, (H, c) = model(X[days_of_years[split_year+1]:],H,c)
    test_loss = criterion(Y[days_of_years[split_year+1]:],Y_hat).item()
    return train_loss, val_loss, test_loss