# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
from cmath import log
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *


def main():

   # ---------------- Paramètres et hyperparamètres ----------------#
    # Forcer l'utilisation du CPU (si un GPU est disponible)
    force_cpu = 1
    training = 0             # Faire l'entrainement sur l'ensemble de donnees
    display_attention = 1       # Affichage des poids d'attention
    display_confusion = 0    # Affichage de la matrice de confusion
    # Visualiser les courbes d'apprentissage pendant l'entrainement
    learning_curves = 1
    test = 1                    # Visualiser la generation sur des echantillons de validation
    batch_size = 100            # Taille des lots
    n_epochs = 50               # Nombre d'iteration sur l'ensemble de donnees
    lr = 0.01                   # Taux d'apprentissage pour l'optimizateur
    save_model = 1              # Sauvegarder le model
    n_hidden = 20               # Nombre de neurones caches par couche
    n_layers = 2               # Nombre de de couches

    n_workers = 0               # Nombre de fils pour charger les donnees
    seed = None                 # Pour repetabilite
    # ------------ Fin des paramètres et hyperparamètres ------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not force_cpu else "cpu")
    print("Device:", device)

    # Instanciation de l'ensemble de données
    data_train_val = HandwrittenWords("data_trainval.p")
    data_test = HandwrittenWords("data_test.p")

    # Séparation de l'ensemble de données (entraînement et validation)
    dataset_train = data_train_val[0:int(len(data_train_val)*0.8)]
    dataset_val = data_train_val[int(len(data_train_val)*0.8):]

    # Instanciation des dataloaders
    dataload_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_test = DataLoader(
        data_test, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    # Instanciation du model
    model = trajectory2seq(device)

    print("nb parameters", sum(p.numel() for p in model.parameters()))

    # Initialisation des variables
    # À compléter

    if training:

        # Initialisation affichage
        if learning_curves:
            train_dist = []  # Historique des distances
            train_loss = []  # Historique des coûts
            val_dist = []  # Historique des distances
            val_loss = []  # Historique des coûts
            fig, ax = plt.subplots(1)  # Initialisation figure

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(
            ignore_index=0)  # ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist = 0

            for batch_idx, data_train_val in enumerate(dataload_train):
                target, input_seq = data_train_val

                # input_seq = input_seq.to(device)

                optimizer.zero_grad()
                # print("input_seq:", input_seq.shape)
                output, hidden, attention = model(input_seq)
                # TODO make it work with target.to(device).long()
                # print("target", target)

                # print("target size", target.size())
                # print("output size", output.size())

                output_view = output.view((-1, model.decoder_dict_size))
                # print("output_view size", output_view.size())

                target_view = target.view((-1)).long()
                # print("target_view size", target_view.size())

                loss = criterion(output_view, target_view)
                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()

                # Calcul de la distance
                output_list = torch.argmax(
                    output, dim=-1).detach().cpu().tolist()

                # TODO make it work with target.to(device).long()
                dist += edit_distance_list(output_list, target)/batch_size

                print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx *
                    batch_size, len(dataload_train.dataset),
                    100. * batch_idx * batch_size /
                    len(dataload_train.dataset), running_loss_train /
                    (batch_idx + 1),
                    dist/len(dataload_train)), end='\r')

            # Validation
            running_loss_val = 0
            dist_val = 0

            for batch_idx, data_train_val in enumerate(dataload_val):
                target, input_seq = data_train_val

                input_seq = input_seq.to(device)

                output, hidden, attention = model(input_seq)

                output_view = output.view((-1, model.decoder_dict_size))
                target_view = target.view((-1)).long()

                loss = criterion(output_view, target_view)
                running_loss_val += loss.item()

                # Calcul de la distance
                output_list = torch.argmax(
                    output, dim=-1).detach().cpu().tolist()

                dist_val += edit_distance_list(output_list, target)/batch_size

            print('Validation - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                epoch, n_epochs, batch_idx *
                batch_size, len(dataload_val.dataset),
                100. * batch_idx * batch_size /
                len(dataload_val.dataset), running_loss_val /
                (batch_idx + 1),
                dist_val/len(dataload_val)))
            # print()
            # print("target 0", dataset.symbols_to_letters(target[0]))
            # print("output 0", dataset.symbols_to_letters(output[0].argmax(dim=-1)))
            matrix = confusion_matrix_batch(output, target)
            # print(matrix)

            # Ajouter les loss aux listes
            # À compléter

            # save model
            if save_model:
                torch.save(model.state_dict(),
                           "model_epoch_{}.pt".format(epoch))

            if learning_curves:
                train_loss.append(running_loss_train/len(dataload_train))
                train_dist.append(dist/len(dataload_train))
                val_loss.append(running_loss_val/len(dataload_val))
                val_dist.append(dist_val/len(dataload_val))
                # plt.figure("learning curves")
                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(train_dist, label='training distance')
                ax.plot(val_loss, label='validation loss')
                ax.plot(val_dist, label='validation distance')
                ax.legend()
                plt.draw()
                plt.pause(0.01)

            if display_confusion:
                plt.figure("Attention")
                ax.cla()
                plt.imshow(matrix)
                plt.ylabel('True class')
                letters_position = list(range(dataset.answer_dict_size))[:-2]
                letters = list(dataset.dictionary.keys())[1:-1]
                plt.yticks(letters_position, letters)
                plt.xticks(letters_position, letters)
                plt.xlabel('Predicted class')
                # plt.colorbar()
                plt.draw()
                plt.pause(0.01)

            # Affichage
        if learning_curves:
            # visualization
            plt.show()
            plt.close('all')

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter

        # Affichage de la matrice de confusion
        # À compléter

        # Évaluation

        # Chargement des poids
        state_dict = torch.load('model_epoch_50.pt')
        model.load_state_dict(state_dict)
        # dataset.symb2int = model.symb2int
        # dataset.int2symb = model.int2symb

        # Affichage des résultats
        for i in range(5):
            # Extraction d'une séquence du dataset de validation
            target, input_seq = data_test[np.random.randint(0, len(data_test))]
            input_padded = input_seq.unsqueeze(0)
            # Évaluation de la séquence
            output, hidden, attention_weights = model(input_padded)
            # attention_weights[2, 2, 2, 2, 2, 2]
            # Affichage
            # in_seq = [model.int2symb['fr'][i] for i in fr_seq.detach().cpu().tolist()]
            # target = [model.int2symb['en'][i] for i in target_seq.detach().cpu().tolist()]
            # out_seq = [model.int2symb['en'][i] for i in out]

            # out_seq = out_seq[:out_seq.index('<eos>')+1]
            # in_seq = in_seq[:in_seq.index('<eos>')+1]
            # target = target[:target.index('<eos>')+1]

            # print('Input:  ', ' '.join(in_seq))
            # print('Target: ', ' '.join(target))
            # print('Output: ', ' '.join(out_seq))
            output_list = dataset.symbols_to_letters(output[0].argmax(dim=-1))
            print("output_string", output_list)
            print('')

            if display_attention:
                plt.figure("Attention")
                for i in range(len(output_list)):
                    plt.subplot(len(output_list), 1, i+1)
                    attn = attention_weights[i]
                    color_map = []
                    for j in range(attn.shape[1]):
                        color_map.append(attn[0][j].item())
                    # plt.plot(attn_x, attn_y, 'o', color='red', markersize=10)
                    # plt.imshow(attn[0:len(input_seq), 0:len(output)], origin='lower',  vmax=1, vmin=0, cmap='pink')

                    plt.yticks([0], [output_list[i]])
                    x_coord = []
                    y_coord = []
                    x_i = 0
                    y_i = 0
                    last_good_index = 0
                    for i in range(input_seq.size(0)):
                        theta = input_seq[i].item()
                        if theta == 6:
                            last_good_index = i
                            break
                        x_i += np.cos(theta)
                        y_i += np.sin(theta)
                        x_coord.append(x_i)
                        y_coord.append(y_i)
                    x_coord_norm = [x / max(x_coord) for x in x_coord]
                    y_coord_norm = [y / max(y_coord) for y in y_coord]
                    plt.scatter(x_coord_norm[0:last_good_index], y_coord_norm[0:last_good_index],  c=color_map[0:last_good_index], marker='o', s=0.5)
                plt.show()


if __name__ == '__main__':
    main()
