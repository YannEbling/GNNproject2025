import numpy as np
import torch
import matplotlib.pyplot as plt

import xv_network as net
from simulation3 import sample_random_incon, create_dataset
import os


def visualize_prediction(x_tensor, lr, hs, nl, ws, model_name, k=1, pre_path=""):
    x = x_tensor.detach().numpy()
    x_vals = x[:, 0]
    y_vals = x[:, 1]

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, color="k", marker="+", label="predicted_trajectory")
    ax.plot([0], [0], color="orange", label="center of mass", marker="o", ls="")
    ax.grid(ls=":", alpha=0.5)
    ax.legend()
    ax.set_title("Prediction of model trained with lr="+str(lr) + ", hs="+str(hs) + ", nl="+str(nl)+ ", ws="+str(ws))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    path = "./plots/"+pre_path+"/"+model_name

    if not os.path.isdir(path):
        os.mkdir(path=path)

    plt.savefig(path+"/prediction_"+str(k)+".png")
    #plt.show()
    plt.close()


def model_predictor(hyperparameters=None, conservation_mode="1", model_name=None, pre_path=""):
    if hyperparameters is None:
        hyperparameters = {"epochs": 101,
                               "lr": 0.01,
                               "batch_size": 1,
                               "pinn_parameter": 0.5,
                               "window_size": 10,
                               "hidden_size": 32,
                               "num_layers": 5,
                               "train_set_size": 1000,
                               "validation_set_size": 200}

    window_size = hyperparameters["window_size"]
    hidden_size = hyperparameters["hidden_size"]
    num_layers = hyperparameters["num_layers"]
    learning_rate = hyperparameters["lr"]

    if model_name is None:
        model_name = "xvpredictor_" + str(hyperparameters["epochs"]) + "ep_" + str(hyperparameters["lr"]) + "lr_" + \
                     str(hyperparameters["pinn_parameter"]) + "pp_" + str(hyperparameters["window_size"]) + "ws_" + \
                     str(hyperparameters["hidden_size"]) + "hs_" + str(hyperparameters["num_layers"]) + "nl_" + \
                     str(hyperparameters["train_set_size"]) + "ts_" + str(hyperparameters["validation_set_size"]) + "vs_"+\
                     str(conservation_mode)+"cm"

    model_name = model_name.replace('.', '_')
    model_name = model_name.replace(',', '_')

    model = net.XVPredictor(window_size=window_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load("./models/"+pre_path+"/"+model_name+".pth.tar", weights_only=False)["model"])
    model.eval()

    for k in range(10):
        T = 1

        initial_data = create_dataset(1, t=np.linspace(0, T/20, model.window_size), r_mean=1, r_std=0., v_mean=2*torch.pi, v_std=0.,
                                      angular_velocity_threshold=0.4)[0].ravel()

        predicted_x = model.predict_trajectory(timepoints=np.linspace(0, T, num=500), x_in=initial_data)
        visualize_prediction(x_tensor=predicted_x, lr=learning_rate, hs=hidden_size, nl=num_layers, ws=window_size,
                             model_name=model_name, k=k)
