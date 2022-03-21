# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

size_x = 64
size_y = 32
nb_symbols = size_x * size_y
dict_size = nb_symbols + 3


class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self):

        # Lecture du text
        self.pad_symbol = pad_symbol = nb_symbols  # '<pad>'
        self.start_symbol = start_symbol = nb_symbols + 1  # '<sos>'
        self.stop_symbol = stop_symbol = nb_symbols + 2  # '<eos>'

        self.word_list_real = dict()
        with open('data_trainval.p', 'rb') as fp:
            self.word_list_real = pickle.load(fp)

        print('Nombre de mots:', len(self.word_list_real))

        word_list_symbols = []

        # Extraction des symboles
        max_sequence_len = 0
        for i, word in enumerate(self.word_list_real):
            positions = word[1]
            all_x = positions[0]
            # normalize from 0 to 1
            all_x_normalized = (all_x - np.min(all_x)) / (np.max(all_x) - np.min(all_x))
            all_y = positions[1]
            all_y_normalized = (all_y - np.min(all_y)) / (np.max(all_y) - np.min(all_y))
            # quantize normalized positions in 64x32 grid
            all_x_quantized = np.round(all_x_normalized * (size_x-1)).astype(int)
            all_y_quantized = np.round(all_y_normalized * (size_y-1)).astype(int)
            # symbol representing position in grid merging x and y
            symbols = all_x_quantized + all_y_quantized * size_x
            if(len(symbols) > max_sequence_len):
                max_sequence_len = len(symbols)
            word_list_symbols.append((word[0], symbols))

        self.word_list_symbols = word_list_symbols
        self.word_list_one_hot = []

        # Insert sos, eos and pad symbols
        for i, word in enumerate(self.word_list_symbols):
            answer = word[0]
            symbols = word[1]
            nb_padding = max_sequence_len - len(symbols)
            symbols = np.insert(symbols, 0, start_symbol)
            symbols = np.insert(symbols, len(symbols), stop_symbol)
            symbols = np.pad(symbols, (0, nb_padding), mode='constant', constant_values=pad_symbol)
            self.word_list_symbols[i] = (word[0], torch.tensor(symbols, dtype=torch.long))

        print("example : ", self.word_list_symbols[0])
        # print("example : ", self.word_list_one_hot[0])

    def __len__(self):
        return len(self.word_list_real)

    def __getitem__(self, idx):
        # return self.data_grid[idx]
        return self.word_list_symbols[idx]

    def visualisation(self, idx):
        # Visualisation des échantillons
        answer = self.word_list_real[idx][0]
        positions = self.word_list_real[idx][1]
        all_x = positions[0]
        all_y = positions[1]

        # show in matplotlib
        # plt.plot(all_x, all_y, 'o')
        # # show text
        # plt.text(all_x[0], all_y[0], answer)

        # data_grid = self.data_grid[idx][1]
        # x_grid = data_grid[0]
        # y_grid = data_grid[1]

        symbols = self.word_list_symbols[idx][1]
        x_from_symbols = symbols % size_x
        y_from_symbols = symbols // size_x

        # print("x_grid", x_grid[0], "y_grid", y_grid[0], "symbol", symbols[0],
        #       "x_from_symbols", x_from_symbols[0], "y_from_symbols", y_from_symbols[0])
        # plot 0 special case
        # plt.plot(x_grid[0], y_grid[0], 'x')
        # show in matplotlib

        # title
        plt.figure(num='This is the title : '+answer)

        # show in subplot 1
        plt.subplot(2, 1, 1)

        # draw line from point to point
        for i in range(len(x_from_symbols)):
            if(i == 0):
                plt.plot(x_from_symbols[i], y_from_symbols[i], 'x')
            else:
                if(symbols[i] < nb_symbols):
                    plt.plot([x_from_symbols[i-1], x_from_symbols[i]],
                             [y_from_symbols[i-1], y_from_symbols[i]], 'b')

        plt.plot(x_from_symbols, y_from_symbols, 'o')

        # show in subplot 2
        plt.subplot(2, 1, 2)
        plt.plot(all_x, all_y, 'o')

        # grid every pixel
        # plt.plot(x_grid, y_grid, 'o')

        plt.text(0, 0, answer)

        plt.show()


if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords()
    for i in range(1):
        a.visualisation(np.random.randint(0, len(a)))
