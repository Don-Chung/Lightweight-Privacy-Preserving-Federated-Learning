# Lightweight-Privacy-Preserving-Federated-Learning
## Description:

Six folders here:

   LPF_Fmnist (Our scheme): It uses Fashion-MNIST data-sets; Including static and dynamic simulations

   LPF_Cifar10 (Our scheme): It uses Cifar10 data-sets; Including static and dynamic simulations

   LPF_notrain (Our scheme): it does not use data-sets to train, we give a dummy model

   LPF_notrain_ty (Ideal FL scheme) : it does not use data-sets to train, we give a dummy model

   LPF_train_ty (Ideal FL scheme): It uses Fashion-MNIST data-sets

   LPF_train_ty_Cifar10 (Ideal FL scheme): It uses Cifar10 data-sets

## Requirements

- torch 1.4.0
- Fashion-MNIST data-sets
- Cifar10 data-sets

## Run

  In each folder, run **RUNME.sh** file.

  If you want to modify the parameters, please modify it in file **para.ini**, it including the number of users, rounds of the protocol, whether to run in static or dynamic mode etc.
