import numpy as np
import torch

import network_training as net_train
import network_prediction as net_pred

hyperparameters = {"epochs": 51,
                           "lr": 0.001,
                           "batch_size": 1,
                           "pinn_parameter": 0.05,
                           "window_size": 5,
                           "hidden_size": 32,
                           "num_layers": 5,
                           "train_set_size": 50,
                           "validation_set_size": 10}

#c_modes = ["", "1", "2", "12"]
c_modes = ["1"]

for conservation_mode in c_modes:
    model_name = "xvpredictor_" + str(hyperparameters["epochs"]) + "ep_" + str(hyperparameters["lr"]) + "lr_" + \
                 str(hyperparameters["pinn_parameter"]) + "pp_" + str(hyperparameters["window_size"]) + "ws_" + \
                 str(hyperparameters["hidden_size"]) + "hs_" + str(hyperparameters["num_layers"]) + "nl_" + \
                 str(hyperparameters["train_set_size"]) + "ts_" + str(hyperparameters["validation_set_size"]) + "vs_" + \
                 str(conservation_mode) + "cm_NONOISE"
    net_train.train_new_model(hyperparameters, conservation_mode=conservation_mode, model_name=model_name)
    net_pred.model_predictor(hyperparameters, conservation_mode=conservation_mode, model_name=model_name)
