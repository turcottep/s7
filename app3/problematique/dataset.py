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
max_len_input = 456
nb_symbols = size_x * size_y
dict_size = nb_symbols + 3

dictionary = {'#': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13,
              'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '$': 27}
# #:sos, $:eos, %:pad
reverse_dictionary = {0: '#', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l',
                      13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '$'}
answer_dict_size = len(dictionary)


class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):

        # Lecture du text
        self.pad_symbol = pad_symbol = nb_symbols  # '<pad>'
        self.start_symbol = start_symbol = nb_symbols + 1  # '<sos>'
        self.stop_symbol = stop_symbol = nb_symbols + 2  # '<eos>'

        self.word_list_real = dict()
        with open(filename, 'rb') as fp:
            self.word_list_real = pickle.load(fp)

        print('Nombre de mots:', len(self.word_list_real))

        self.word_list_symbols = []
        word_list_symbols = []

        # Transformation de (x,y) en indice de la case dans la grille
        # [1,2,3
        # 4,5,6] grille 64x32 aka indice 0-2047
        self.max_sequence_len = 0
        max_answer_len = 0
        for i, word in enumerate(self.word_list_real):
            answer = word[0]
            positions = word[1]

            # normalize position from 0 to 1
            all_x = positions[0]
            all_y = positions[1]

            all_x_normalized = (all_x - np.min(all_x)) / (np.max(all_x) - np.min(all_x))
            all_y_normalized = (all_y - np.min(all_y)) / (np.max(all_y) - np.min(all_y))

            all_x_to_next_x = []
            for j in range(len(all_x_normalized) - 1):
                all_x_to_next_x.append(all_x_normalized[j+1] - all_x_normalized[j])
            all_y_to_next_y = []
            for j in range(len(all_y_normalized) - 1):
                all_y_to_next_y.append(all_y_normalized[j+1] - all_y_normalized[j])

            all_angles = []
            for j in range(len(all_x_to_next_x)):
                all_angles.append(np.arctan2(all_y_to_next_y[j], all_x_to_next_x[j]))

            # quantize normalized positions in 64x32 grid
            # all_x_quantized = np.round(all_x_normalized * (size_x-1)).astype(int)
            # all_y_quantized = np.round(all_y_normalized * (size_y-1)).astype(int)

            # # symbol representing position in grid merging x and y
            # symbols = all_x_quantized + all_y_quantized * size_x

            # get max sequence length
            if(len(all_x_to_next_x) > self.max_sequence_len):
                self.max_sequence_len = len(all_x_to_next_x)

            # get max answer length
            if(len(answer) > max_answer_len):
                max_answer_len = len(answer)

            # add to list of words
            word_list_symbols.append((answer, all_angles))

        # Insert sos, eos and pad symbols
        for i, word in enumerate(word_list_symbols):
            answer = word[0]
            symbols = word[1]
            if answer == []:
                answer = ""
            # add sos,eos,pad to words
            nb_pad = max_answer_len - len(answer)
            answer = answer + "$" + nb_pad * "#"

            # get answer in symbol (0-27) form, and one-hot
            # answer_symbol = torch.zeros(max_answer_len + 2, len(dictionary))
            # for j, letter in enumerate(answer):
            #     letter_symbol = dictionary[letter]
            #     answer_symbol[j][letter_symbol] = 1

            answer_symbol = torch.zeros(max_answer_len + 1)
            for j, letter in enumerate(answer):
                letter_symbol = dictionary[letter]
                answer_symbol[j] = letter_symbol

            # all_x = symbols[0]
            # all_y = symbols[1]

            nb_padding = max_len_input - len(symbols)

            # symbols = np.insert(symbols, len(symbols), stop_symbol)

            all_x_paded = np.pad(all_x, (0, nb_padding), 'constant', constant_values=0)
            # all_y_paded = np.pad(all_y, (0, nb_padding), 'constant', constant_values=0)
            # symbols = torch.tensor([all_x_paded, all_y_paded], dtype=torch.float)
            symbols = np.pad(symbols, (0, nb_padding), 'constant', constant_values=6)
            symbols = torch.tensor(symbols, dtype=torch.float)
            self.word_list_symbols.append((answer_symbol, symbols))

        print("example : ", self.word_list_symbols[0])
        print("answer_symbol : ", answer_symbol)
        # self.from_onehot_to_letter(self.word_list_symbols[5][0])
        # print("example : ", self.word_list_one_hot[0])

    def __len__(self):
        return len(self.word_list_real)

    def from_onehot_to_letter(self, answer_symbol):
        data = []
        for i, onehotvector in enumerate(answer_symbol):
            a = onehotvector.argmax().item()
            print("argmax", a)
            b = list(dictionary)[a]
            print("b", b)
            data.append(b)
        print("from_onehot_to_letter", data)
        return data

    def __getitem__(self, idx):
        # return self.data_grid[idx]
        return self.word_list_symbols[idx]

    def get_raw_input(self, idx):
        return self.word_list_real[idx]

    def visualisation(self, nb_examples=1):
        # Visualisation des échantillons
        for idx in range(nb_examples):

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
            x_delta = symbols[0]
            y_delta = symbols[1]

            x_coord = []
            y_coord = []
            x_i = 0
            y_i = 0
            for i in range(len(x_delta)):
                x_i += x_delta[i].item()
                y_i += y_delta[i].item()
                x_coord.append(x_i)
                y_coord.append(y_i)

            # print("x_coord : ", x_coord)
            # print("x_grid", x_grid[0], "y_grid", y_grid[0], "symbol", symbols[0],
            #       "x_from_symbols", x_from_symbols[0], "y_from_symbols", y_from_symbols[0])
            # plot 0 special case
            # plt.plot(x_grid[0], y_grid[0], 'x')
            # show in matplotlib

            # title
            plt.figure(num='This is the title : '+answer)

            # show in subplot 1
            plt.subplot(2, 1, 1)

            plt.plot(x_coord, y_coord, 'o')

            # plt.plot(x_from_symbols, y_from_symbols, 'o')

            # show in subplot 2
            plt.subplot(2, 1, 2)
            plt.plot(all_x, all_y, 'o')

            # grid every pixel
            # plt.plot(x_grid, y_grid, 'o')

            plt.text(0, 0, answer)

            plt.show()


def symbols_to_letters(symbols):
    letters = []
    for symbol in symbols:
        letter = reverse_dictionary[symbol.item()]
        letters.append(letter)
    return letters


if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords()
    a.visualisation(10)
