#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset,random_split
from datasets import GlucoseData
from train import train_model
from radam import RAdam
from optimizer import Lookahead
from ranger import Ranger
import numpy as np
from torch.utils.data.sampler import  SubsetRandomSampler
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=10, hidden_size=40):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_dim=10, output_dim=8, hidden_size=42):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
        
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN()
        self.decoder = DecoderRNN()


    def forward(self, input, hiddenIn):
        input = input.view(70, 400, -1)
        output1, hidden1 = self.encoder(input, hiddenIn)
        output, hidden = self.decoder(output1, hidden1)

        return output, hidden

