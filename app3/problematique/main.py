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
    training = 1                # Faire l'entrainement sur l'ensemble de donnees
    display_attention = 0       # Affichage des poids d'attention
    # Visualiser les courbes d'apprentissage pendant l'entrainement
    learning_curves = 1
    test = 1                    # Visualiser la generation sur des echantillons de validation
    batch_size = 100            # Taille des lots
    n_epochs = 100               # Nombre d'iteration sur l'ensemble de donnees
    lr = 0.01                   # Taux d'apprentissage pour l'optimizateur

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
    data = HandwrittenWords()

    # Séparation de l'ensemble de données (entraînement et validation)
    dataset_train = data[0:int(len(data)*0.8)]
    dataset_val = data[int(len(data)*0.8):]

    # Instanciation des dataloaders
    dataload_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, num_workers=n_workers)

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

            for batch_idx, data in enumerate(dataload_train):
                target, input_seq = data

                input_seq = input_seq.to(device)

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

            for batch_idx, data in enumerate(dataload_val):
                target, input_seq = data

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
            print()
            print("target 0", dataset.symbols_to_letters(target[0]))
            print("output 0", dataset.symbols_to_letters(output[0].argmax(dim=-1)))

            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            # À compléter

            if learning_curves:
                train_loss.append(running_loss_train/len(dataload_train))
                train_dist.append(dist/len(dataload_train))
                val_loss.append(running_loss_val/len(dataload_val))
                val_dist.append(dist_val/len(dataload_val))
                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(train_dist, label='training distance')
                ax.plot(val_loss, label='validation loss')
                ax.plot(val_dist, label='validation distance')
                ax.legend()
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

        pass


if __name__ == '__main__':
    main()
