import numpy as np
import torch

import network_training as net_train
import network_prediction as net_pred

hyperparameters = {"epochs": 101,
                           "lr": 0.001,
                           "batch_size": 1,
                           "pinn_parameter": 0.2,
                           "window_size": 10,
                           "hidden_size": 32,
                           "num_layers": 5,
                           "train_set_size": 1000,
                           "validation_set_size": 200}

c_modes = ["", "1", "2", "12"]

for conservation_mode in c_modes:
    net_train.main(hyperparameters, conservation_mode=conservation_mode)
    net_pred.model_predictor(hyperparameters, conservation_mode=conservation_mode)
