# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from dataset import *


class trajectory2seq(nn.Module):
    def __init__(self):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        # self.hidden_dim = n_hidden
        # self.n_layers = n_layers
        # self.device = device
        # self.symb2int = symb2int
        # self.int2symb = int2symb
        # self.dict_size = dict_size
        # self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn
        hidden_dim = 256
        n_layers = 2
        self.rnn = nn.RNN(input_size=dict_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)

        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        # À compléter

    def forward(self, x):
        # À compléter
        print(x.shape)
        words_one_hot = torch.zeros(x.shape[0], x.shape[1], dict_size)
        for i, word in enumerate(x):
            for j, symbol in enumerate(word):
                words_one_hot[i][j][symbol] = 1

        # print("", words_one_hot[0])
        # print("words one hot shape", words_one_hot.shape)
        out = self.rnn(words_one_hot)

        return out, None, None
