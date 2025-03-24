import xv_network as net
import simulation3 as sim

import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def save_checkpoint(state, filename="/models/new_model.pth.tar"):
    print("=> Saving checkpoint!")
    torch.save(state, filename)


def train_new_model(hyperparameters=None, conservation_mode="1", model_name=None, random_t=False):
    if hyperparameters is None:
        hyperparameters = {"epochs": 5,
                           "lr": 0.01,
                           "batch_size": 1,
                           "pinn_parameter": 0.0,
                           "window_size": 10,
                           "hidden_size": 32,
                           "num_layers": 5,
                           "train_set_size": 300,
                           "validation_set_size": 100}

    train_set_size = hyperparameters["train_set_size"]
    validation_set_size = hyperparameters["validation_set_size"]
    window_size = hyperparameters["window_size"]
    hidden_size = hyperparameters["hidden_size"]
    num_layers = hyperparameters["num_layers"]

    trajectory_model = net.XVPredictor(window_size=window_size, hidden_size=hidden_size, num_layers=num_layers)

    train_set = torch.tensor(sim.create_dataset(train_set_size,
                                                r_mean=1, r_std=0, v_mean=2*np.pi, v_std=0,
                                                angular_velocity_threshold=0.4, random_t=random_t),
                             dtype=torch.float32)
    validation_set = torch.tensor(sim.create_dataset(validation_set_size,
                                                     r_mean=1, r_std=0, v_mean=2*np.pi, v_std=0,
                                                     angular_velocity_threshold=0.4, random_t=random_t),
                                  dtype=torch.float32)

    train_loss, validation_loss = net.train_xv_predictor(train_set, validation_set, trajectory_model,
                                                         hyperparameters=hyperparameters,
                                                         conservation_mode=conservation_mode)

    num_epochs = np.arange(hyperparameters["epochs"])
    num_vali_epochs = 10 * np.arange((hyperparameters["epochs"]-1)//10+1)
    mean_train_loss = np.zeros_like(num_epochs)
    mean_vali_loss = np.zeros_like(num_vali_epochs)

    print(train_loss)
    print(validation_loss)

    for k in range(len(train_loss)):
        mean_train_loss[k] = np.mean(train_loss[k])
    for k in range(len(validation_loss)):
        mean_vali_loss[k] = np.mean(validation_loss[k])


    model_dict = {"model": trajectory_model.state_dict(),
                  "hyperparameters": hyperparameters,
                  "validation_epochs": num_vali_epochs,
                  "validation_loss": validation_loss,
                  "training_epochs": num_epochs,
                  "training_loss": train_loss}

    if model_name is None:
        model_name = "xvpredictor_"+str(hyperparameters["epochs"])+"ep_"+str(hyperparameters["lr"])+"lr_"+\
                     str(hyperparameters["pinn_parameter"])+"pp_"+str(hyperparameters["window_size"])+"ws_"+\
                     str(hyperparameters["hidden_size"])+"hs_"+str(hyperparameters["num_layers"])+"nl_"+\
                     str(hyperparameters["train_set_size"])+"ts_"+str(hyperparameters["validation_set_size"])+"vs_"+\
                     str(conservation_mode)+"cm"

    model_name = model_name.replace('.', '_')
    model_name = model_name.replace(',', '_')

    save_checkpoint(model_dict, "./models/" + model_name + ".pth.tar")
    print("Saved a model!")

    plt.figure(figsize=(8, 6))

    plt.plot(num_epochs, mean_train_loss, ls=":", color="red", marker="+", label="train loss over epochs", alpha=0.6)
    plt.plot(num_vali_epochs, mean_vali_loss, ls="-", color="blue", marker="+", label="validation loss over epochs")

    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.title("Loss evolution during training")
    plt.grid(alpha=0.6)
    plt.legend()

    plt.savefig("./plots/"+model_name+"_loss_evolution.png")

    plt.show()
    plt.close()


def train_existing_model(hyperparameters=None, conservation_mode="1", model_name=None, random_t=False):
    if hyperparameters is None:
        hyperparameters = {"epochs": 5,
                           "lr": 0.01,
                           "batch_size": 1,
                           "pinn_parameter": 0.0,
                           "window_size": 10,
                           "hidden_size": 32,
                           "num_layers": 5,
                           "train_set_size": 300,
                           "validation_set_size": 100}


    train_set_size = hyperparameters["train_set_size"]
    validation_set_size = hyperparameters["validation_set_size"]
    window_size = hyperparameters["window_size"]
    hidden_size = hyperparameters["hidden_size"]
    num_layers = hyperparameters["num_layers"]

    if not os.path.isfile("./models/" + model_name + ".pth.tar"):
        print("No model with the specified name found in the ./models/ directory.")
        return -1
    else:
        model = net.XVPredictor(window_size=window_size, hidden_size=hidden_size, num_layers=num_layers)
        model.load_state_dict(torch.load("./models/" + model_name + ".pth.tar", weights_only=False)["model"])
        old_epochs = torch.load("./models/" + model_name + ".pth.tar", weights_only=False)["training_epochs"]

    train_set = torch.tensor(sim.create_dataset(train_set_size,
                                                r_mean=1, r_std=0, v_mean=2*np.pi, v_std=0,
                                                angular_velocity_threshold=0.4, random_t=random_t),
                             dtype=torch.float32)
    validation_set = torch.tensor(sim.create_dataset(validation_set_size,
                                                     r_mean=1, r_std=0, v_mean=2*np.pi, v_std=0,
                                                     angular_velocity_threshold=0.4, random_t=random_t),
                                  dtype=torch.float32)

    train_loss, validation_loss = net.train_xv_predictor(train_set, validation_set, model,
                                                         hyperparameters=hyperparameters,
                                                         conservation_mode=conservation_mode)

    num_epochs = np.arange(hyperparameters["epochs"])
    num_vali_epochs = 10 * np.arange((hyperparameters["epochs"]-1)//10+1)
    mean_train_loss = np.zeros_like(num_epochs)
    mean_vali_loss = np.zeros_like(num_vali_epochs)


    for k in range(len(train_loss)):
        mean_train_loss[k] = np.mean(train_loss[k])
    for k in range(len(validation_loss)):
        mean_vali_loss[k] = np.mean(validation_loss[k])


    model_dict = {"model": model.state_dict(),
                  "hyperparameters": hyperparameters,
                  "validation_epochs": num_vali_epochs,
                  "validation_loss": validation_loss,
                  "training_epochs": num_epochs+old_epochs,
                  "training_loss": train_loss}

    save_checkpoint(model_dict, "./models/" + model_name + ".pth.tar")
    print("Saved a model!")

    plt.figure(figsize=(8, 6))

    plt.plot(num_epochs, mean_train_loss, ls=":", color="red", marker="+", label="train loss over epochs", alpha=0.6)
    plt.plot(num_vali_epochs, mean_vali_loss, ls="-", color="blue", marker="+", label="validation loss over epochs")

    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.title("Loss evolution during training")
    plt.grid(alpha=0.6)
    plt.legend()

    plt.savefig("./plots/" + model_name + "_loss_evolution.png")

    plt.show()
    plt.close()


if __name__ == "__main__":
    train_new_model()
