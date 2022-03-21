# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import numpy as np

dictionary = {'#': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13,
              'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26}


def edit_distance(x, y):
    x = x.lower()
    x = "#" + x
    y = y.lower()
    y = "#" + y

    x_list = list(x)
    y_list = list(y)
    x_list_int = []
    y_list_int = []
    for i in range(len(x_list)):
        x_list_int.append(dictionary[x_list[i]])
    for i in range(len(y_list)):
        y_list_int.append(dictionary[y_list[i]])

    x_array = np.array(x_list_int)
    y_array = np.array(y_list_int)

    D = np.zeros((len(x_array), len(y_array)))
    D[:, 0] = np.arange(len(x_array))
    D[0, :] = np.arange(len(y_array))
    for i in range(1, len(x_array)):
        for j in range(1, len(y_array)):
            opt1 = D[i - 1, j] + 1
            opt2 = D[i, j - 1] + 1
            opt3 = D[i - 1, j - 1] + (x_array[i] != y_array[j])
            D[i, j] = min(opt1, opt2, opt3)

    print(D)
    distance = D[len(x_array) - 1, len(y_array) - 1]

    return distance


def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion

    # À compléter

    return None


if __name__ == "__main__":
    word1 = "back"
    word2 = "hititfromtheback"
    print("Distance d'édition:", edit_distance(word1, word2))
