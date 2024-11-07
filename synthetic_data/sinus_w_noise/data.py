import numpy as np
import matplotlib.pyplot as plt
import torch


class Data(): 
    def __init__(self, p, T, tau=0): #Create sinus wave of length T with period 2pi/p with noise tau
        self.time = torch.arange(0, T, dtype=torch.float32)
        self.x = torch.sin(p* self.time) + torch.randn(T) * tau

    def __getitem__(self,N): #Create data chunks: returns tens[x_i,...,x_{i+N}], tens[x_N] for each i in T. tens shape(1000-N,N), (1000-N)
        in_data = [self.x[i: self.T-N+i] for i in range(N)]
        in_data = torch.stack(in_data,1).reshape(self.T-N,N)
        out_data = self.x[N:].reshape(self.T-N,1)
        return in_data, out_data       