# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import numpy as np

import dataset


def edit_distance_list(x_list, y_list):
    # Calcul de la distance d'édition entre une liste de mots et une liste de mots

    distance = 0
    for x, y in zip(x_list, y_list):
        distance += edit_distance(x, y)

    return distance


def edit_distance(x, y):

    D = np.zeros((len(x), len(y)))
    D[:, 0] = np.arange(len(x))
    D[0, :] = np.arange(len(y))
    for i in range(1, len(x)):
        for j in range(1, len(y)):
            opt1 = D[i - 1, j] + 1
            opt2 = D[i, j - 1] + 1
            opt3 = D[i - 1, j - 1] + (x[i] != y[j])
            D[i, j] = min(opt1, opt2, opt3)

    # print(D)
    distance = D[len(x) - 1, len(y) - 1]
    # print("distance", distance)
    return distance


def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion

    # À compléter

    return None


if __name__ == "__main__":
    word1 = "back"
    word2 = "hititfromtheback"

    x = x.word1()
    x = "#" + x
    y = y.word2()
    y = "#" + y

    x_list = list(x)
    y_list = list(y)
    x_list_int = []
    y_list_int = []
    for i in range(len(x_list)):
        x_list_int.append(dataset.dictionary[x_list[i]])
    for i in range(len(y_list)):
        y_list_int.append(dataset.dictionary[y_list[i]])

    x_array = np.array(x_list_int)
    y_array = np.array(y_list_int)

    print("Distance d'édition:", edit_distance(x_array, y_array))
