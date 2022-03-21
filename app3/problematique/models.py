# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

import dataset


class trajectory2seq(nn.Module):
    def __init__(self, device):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        # self.hidden_dim = 20
        # self.n_layers = n_layers
        # self.device = device
        # self.symb2int = symb2int
        # self.int2symb = int2symb
        # self.dict_size = dict_size
        # self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn
        n_hidden = 20
        n_layers = 2
        trajectory_dict_size = dataset.dict_size
        answer_dict_size = dataset.answer_dict_size

        self.fr_embedding = nn.Embedding(trajectory_dict_size, n_hidden)
        self.en_embedding = nn.Embedding(answer_dict_size, n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Couches pour attention
        self.att_combine = nn.Linear(2*n_hidden, n_hidden)
        self.hidden2query = nn.Linear(n_hidden, n_hidden)
        # À compléter

        # Couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, answer_dict_size)
        self.to(device)

    def encoder(self, x):
        # Encodeur L2Q4
        out = None
        hidden = None

        return out, hidden

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)

        # Attention
        # L2Q4

        attention_weights = None
        attention_output = None

        return attention_output, attention_weights

    def decoderWithAttn(self, encoder_outs, hidden):
        # Décodeur avec attention

        # Initialisation des variables
        max_len = self.max_len['en']  # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(self.device)  # Vecteur de sortie du décodage
        attention_weights = torch.zeros((batch_size, self.max_len['fr'], self.max_len['en'])).to(self.device)  # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

            vec_out = vec_out

            # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return vec_out, hidden, attention_weights

    def forward(self, x):
        # À compléter
        print(x.shape)
        words_one_hot = torch.zeros(x.shape[0], x.shape[1], dict_size)
        for i, word in enumerate(x):
            for j, symbol in enumerate(word):
                words_one_hot[i][j][symbol] = 1

        # print("", words_one_hot[0])
        # print("words one hot shape", words_one_hot.shape)

        # Passe avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoderWithAttn(out, h)
        return out, hidden, attn
