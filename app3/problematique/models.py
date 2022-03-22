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
        self.n_hidden = n_hidden = 20
        # self.n_layers = n_layers
        self.device = device
        self.max_len_input = dataset.max_len_input
        # self.symb2int = symb2int
        # self.int2symb = int2symb
        # self.dict_size = dict_size
        # self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn

        n_layers = 2
        self.encoder_dict_size = dataset.dict_size
        self.decoder_dict_size = dataset.answer_dict_size

        # encoder
        # self.encoder_embedding = nn.Embedding(self.encoder_dict_size, n_hidden)
        self.encoder_layer = nn.GRU(self.max_len_input, n_hidden, n_layers, batch_first=True)

        # decoder
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_embedding = nn.Embedding(self.decoder_dict_size, n_hidden)

        # Couches pour attention
        self.att_combine = nn.Linear(2*n_hidden, n_hidden)
        self.hidden2query = nn.Linear(n_hidden, n_hidden)
        # À compléter

        # Couche dense pour la sortie
        self.linear_out = nn.Linear(n_hidden, self.decoder_dict_size)
        self.to(self.device)

    def attentionModule(self, hidden, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(hidden)

        attention = torch.bmm(query, torch.permute(values, (0, 2, 1)))
        attention_weights = torch.softmax(attention[:, 0, :], dim=1)
        attention_weights_repeated = attention_weights[:, :, None].repeat(1, 1, self.n_hidden)
        attention_output = torch.sum(attention_weights_repeated * values, dim=1)

        return attention_output, attention_weights

    def decoderWithAttn(self, encoded, hidden):
        # Décodeur avec attention

        # Initialisation des variables
        # max_len_encoder = self.max_len_input
        max_len_decoder = 6
        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len_decoder, self.decoder_dict_size))  # Vecteur de sortie du décodage
        attention_weights = []  # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len_decoder):

            embedded = self.decoder_embedding(vec_in)
            buffer, hidden = self.decoder_layer(embedded, hidden)
            a, attention_weights_buffer = self.attentionModule(buffer, encoded)
            attention_weights.append(attention_weights_buffer)
            buffer2 = self.att_combine(torch.cat((buffer[:, 0, :], a), dim=1))
            temp_temp_boy = self.linear_out(buffer2)
            temp_boy = temp_temp_boy[:, :]
            vec_out[:, i, :] = temp_boy
            vec_in = torch.argmax(vec_out[:, i:i+1, :], dim=2)

        return vec_out, hidden, attention_weights

    def forward(self, x):
        # À compléter
        # print("x_shape", x.shape)
        # words_one_hot = nn.functional.one_hot(x, self.encoder_dict_size)

        # print("words one hot", words_one_hot)
        # print("words one hot shape", words_one_hot.shape)

        # Passe avant
        # embedded = self.encoder_embedding(x)

        encoded, hidden = self.encoder_layer(x)
        out, hidden, attention_weights = self.decoderWithAttn(encoded, hidden)
        # decoded, hidden = self.decoder_layer(encoded)
        # out = self.decoder_embedding(decoded)

        return out, hidden, attention_weights
