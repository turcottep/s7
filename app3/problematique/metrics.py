# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
from matplotlib import pyplot as plt
import numpy as np
import torch

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


def confusion_matrix_batch(predictions, labels):
    matrix = np.zeros((dataset.answer_dict_size, dataset.answer_dict_size))
    for i in range(predictions.shape[0]):
        matrix = confusion_matrix(matrix, predictions[i], labels[i])
    # return rows 1-27 and columns 1-27
    return matrix[1:dataset.answer_dict_size-1, 1:dataset.answer_dict_size-1]


def confusion_matrix(matrix, pred, true):
    # print("pred", pred)
    # print("true", true)
    pred_softmax = torch.softmax(pred, dim=1)
    # print("pred_softmax", pred_softmax)
    for i in range(pred.shape[0]):
        true_i = true[i].long().item()
        for j in range(pred.shape[1]):
            pred_j = pred_softmax[i, j].item()
            # print("true_i", true_i, "pred_j", pred_j)
            matrix[true_i][j] += pred_j
    return matrix


if __name__ == "__main__":
    # word1 = "back"
    # word2 = "hititfromtheback"

    # x = x.word1()
    # x = "#" + x
    # y = y.word2()
    # y = "#" + y

    # x_list = list(x)
    # y_list = list(y)
    # x_list_int = []
    # y_list_int = []
    # for i in range(len(x_list)):
    #     x_list_int.append(dataset.dictionary[x_list[i]])
    # for i in range(len(y_list)):
    #     y_list_int.append(dataset.dictionary[y_list[i]])

    # x_array = np.array(x_list_int)
    # y_array = np.array(y_list_int)

    # print("Distance d'édition:", edit_distance(x_array, y_array))
    pred = torch.tensor([[-2.0676,  1.4447,  0.6858,  0.8045,  0.2711,  1.0849,  0.3380,  0.2880,
                          0.1162,  1.0815, -0.7058, -0.7512,  0.9895,  0.6377,  0.3065,  1.3206,
                          0.2907, -1.3581,  1.2048,  0.3385,  0.4586,  1.0174, -0.4552,  0.0057,
                          -1.6314, -1.1056, -1.5531, -1.1118],
                         [-2.2657,  1.5615,  0.7042,  0.7723,  0.2719,  1.2214,  0.2340,  0.2315,
                          0.0930,  1.2616, -0.8846, -0.7659,  1.0161,  0.6449,  0.4099,  1.3756,
                          0.3298, -1.5331,  1.2400,  0.3169,  0.4552,  1.0870, -0.5486,  0.0223,
                          -1.7716, -1.1313, -1.7001, -1.1360],
                         [-2.7861,  1.4678,  0.4362,  0.4329,  0.3517,  1.4056, -0.1194,  0.0595,
                          -0.0848,  1.1929, -1.4624, -0.5924,  0.9603,  0.4093,  0.5935,  1.1447,
                          0.1471, -2.0957,  1.0887,  0.6025,  0.4850,  0.7365, -0.8135, -0.1737,
                          -2.0501, -0.8588, -1.8587, -0.3964],
                         [-3.5957,  1.1660, -0.1124, -0.1689,  0.5208,  1.5941, -0.6896, -0.1871,
                          -0.3843,  0.8553, -2.3903, -0.2527,  0.8185, -0.0616,  0.8267,  0.6500,
                          -0.2435, -3.0119,  0.7492,  1.2044,  0.5399, -0.0338, -1.2226, -0.5780,
                          -2.4388, -0.3149, -2.0013,  1.0878],
                         [-4.4631,  0.5654, -0.9194, -1.0510,  0.7104,  1.6329, -1.4157, -0.5630,
                          -0.8054,  0.2077, -3.4802,  0.2129,  0.5644, -0.7683,  1.0811, -0.1663,
                          -0.8298, -4.0542,  0.1462,  1.9858,  0.6373, -1.2708, -1.7321, -1.1455,
                          -2.7065,  0.5470, -2.0121,  3.3006],
                         [-5.1041, -0.0369, -1.5911, -1.7914,  0.8756,  1.5788, -1.9777, -0.9368,
                          -1.1365, -0.4706, -4.3211,  0.6221,  0.3323, -1.3758,  1.2662, -0.9450,
                          -1.3692, -4.8708, -0.4138,  2.6575,  0.7361, -2.4391, -2.0961, -1.6614,
                          -2.8094,  1.3458, -1.9538,  5.2965]])
    true = torch.tensor([3., 21., 18., 27.,  0.,  0.])
    matrix = confusion_matrix(pred, true)
    plt.figure()
    plt.imshow(matrix)
    plt.ylabel('True class')
    plt.yticks(range(dataset.answer_dict_size), dataset.dictionary.keys())
    plt.xticks(range(dataset.answer_dict_size), dataset.dictionary.keys())
    plt.xlabel('Predicted class')
    plt.colorbar()
    plt.show()
