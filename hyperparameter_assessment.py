import network_training as train
import network_prediction as pred

import torch
import numpy as np
import matplotlib.pyplot as plt


conservation_mode = "1"

standard_hs = 32
standard_nl = 5
standard_lr = 0.001
standard_ne = 201
standard_pp = 0.001
standard_bs = 128
standard_ts = 1280
standard_vs = 512

hidden_sizes = [8, 16, 32, 64, 128]
num_layers = [1, 2, 3, 4, 5, 6, 7, 8]
learning_rates = [0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
epoch_numbers = [11, 51, 101, 201, 501, 1001, 2001]
pinn_parameter = [1, 0.1, 0.01, 0.001, 0]


def standard_hyperparameters():
    hyperparameters = {"epochs": standard_ne,
                       "lr": standard_lr,
                       "batch_size": standard_bs,
                       "pinn_parameter": standard_pp,
                       "window_size": 1,
                       "hidden_size": standard_hs,
                       "num_layers": standard_nl,
                       "train_set_size": standard_ts,
                       "validation_set_size": standard_vs}
    return hyperparameters


hyperparameters = standard_hyperparameters()
pre_path = "hyperparameter_assessment"

for hs in hidden_sizes:
    if hs == 8 or hs == 16:
        print("Skipping previously trained model...")
        continue
    hyperparameters["hidden_size"] = hs

    model_name = "xvpredictor_hyperparameters_assessment_" + str(hyperparameters["epochs"]) + "ep_" + str(hyperparameters["lr"]) + "lr_" + \
                 str(hyperparameters["pinn_parameter"]) + "pp_" + str(hyperparameters["window_size"]) + "ws_" + \
                 str(hyperparameters["hidden_size"]) + "hs_" + str(hyperparameters["num_layers"]) + "nl_" + \
                 str(hyperparameters["train_set_size"]) + "ts_" + str(hyperparameters["validation_set_size"]) + "vs_" + \
                 str(conservation_mode) + "cm_NONOISE"
    train.train_new_model(hyperparameters, conservation_mode=conservation_mode, model_name=model_name, pre_path=pre_path)
    pred.model_predictor(hyperparameters, conservation_mode=conservation_mode, model_name=model_name, pre_path=pre_path)

hyperparameters = standard_hyperparameters()

for nl in num_layers:
    hyperparameters["num_layers"] = nl

    model_name = "xvpredictor_hyperparameters_assessment_" + str(hyperparameters["epochs"]) + "ep_" + str(
        hyperparameters["lr"]) + "lr_" + \
                 str(hyperparameters["pinn_parameter"]) + "pp_" + str(hyperparameters["window_size"]) + "ws_" + \
                 str(hyperparameters["hidden_size"]) + "hs_" + str(hyperparameters["num_layers"]) + "nl_" + \
                 str(hyperparameters["train_set_size"]) + "ts_" + str(hyperparameters["validation_set_size"]) + "vs_" + \
                 str(conservation_mode) + "cm_NONOISE"
    train.train_new_model(hyperparameters, conservation_mode=conservation_mode, model_name=model_name,
                          pre_path=pre_path)
    pred.model_predictor(hyperparameters, conservation_mode=conservation_mode, model_name=model_name, pre_path=pre_path)

hyperparameters = standard_hyperparameters()

for lr in learning_rates:
    hyperparameters["lr"] = lr

    model_name = "xvpredictor_hyperparameters_assessment_" + str(hyperparameters["epochs"]) + "ep_" + str(
        hyperparameters["lr"]) + "lr_" + \
                 str(hyperparameters["pinn_parameter"]) + "pp_" + str(hyperparameters["window_size"]) + "ws_" + \
                 str(hyperparameters["hidden_size"]) + "hs_" + str(hyperparameters["num_layers"]) + "nl_" + \
                 str(hyperparameters["train_set_size"]) + "ts_" + str(hyperparameters["validation_set_size"]) + "vs_" + \
                 str(conservation_mode) + "cm_NONOISE"
    train.train_new_model(hyperparameters, conservation_mode=conservation_mode, model_name=model_name,
                          pre_path=pre_path)
    pred.model_predictor(hyperparameters, conservation_mode=conservation_mode, model_name=model_name, pre_path=pre_path)

hyperparameters = standard_hyperparameters()

for pp in pinn_parameter:
    if pp == 1:
        print("Skipping previously trained model!")
        continue
    hyperparameters["pinn_parameter"] = pp

    model_name = "xvpredictor_hyperparameters_assessment_" + str(hyperparameters["epochs"]) + "ep_" + str(
        hyperparameters["lr"]) + "lr_" + \
                 str(hyperparameters["pinn_parameter"]) + "pp_" + str(hyperparameters["window_size"]) + "ws_" + \
                 str(hyperparameters["hidden_size"]) + "hs_" + str(hyperparameters["num_layers"]) + "nl_" + \
                 str(hyperparameters["train_set_size"]) + "ts_" + str(hyperparameters["validation_set_size"]) + "vs_" + \
                 str(conservation_mode) + "cm_NONOISE"
    train.train_new_model(hyperparameters, conservation_mode=conservation_mode, model_name=model_name,
                          pre_path=pre_path)
    pred.model_predictor(hyperparameters, conservation_mode=conservation_mode, model_name=model_name, pre_path=pre_path)

hyperparameters = standard_hyperparameters()


hyperparameters["epochs"] = 2001

model_name = "xvpredictor_hyperparameters_assessment_" + str(hyperparameters["epochs"]) + "ep_" + str(
        hyperparameters["lr"]) + "lr_" + \
                 str(hyperparameters["pinn_parameter"]) + "pp_" + str(hyperparameters["window_size"]) + "ws_" + \
                 str(hyperparameters["hidden_size"]) + "hs_" + str(hyperparameters["num_layers"]) + "nl_" + \
                 str(hyperparameters["train_set_size"]) + "ts_" + str(hyperparameters["validation_set_size"]) + "vs_" + \
                 str(conservation_mode) + "cm_NONOISE"
train.train_new_model(hyperparameters, conservation_mode=conservation_mode, model_name=model_name,
                          pre_path=pre_path)
pred.model_predictor(hyperparameters, conservation_mode=conservation_mode, model_name=model_name, pre_path=pre_path)

hyperparameters = standard_hyperparameters()