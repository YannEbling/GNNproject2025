import matplotlib.pyplot as plt
import torch
import numpy as np

def load_loss(path):
    state = torch.load(path, weights_only=False)
    train_epochs = state["training_epochs"]
    train_loss = np.mean(state["training_loss"], axis=-1)
    vali_epochs = state["validation_epochs"]
    vali_loss = np.mean(state["validation_loss"], axis=-1)

    return train_epochs, train_loss, vali_epochs, vali_loss

pre_path = "./models/hyperparameter_assessment/"

# Assessment of hidden size

training_losses = []
validation_losses = []

hidden_sizes = [8, 16, 32, 64, 128]
colors = ["red", "orange", "yellow", "green", "blue"]
for hs in hidden_sizes:
    name = "xvpredictor_hyperparameters_assessment_201ep_0_001lr_0_001pp_1ws_"+str(hs)+"hs_5nl_1280ts_512vs_1cm_NONOISE.pth.tar"
    t_epochs, t_loss, v_epochs, v_loss = load_loss(pre_path+name)
    training_losses.append(t_loss)
    validation_losses.append(v_loss)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title("Training loss for different hidden sizes")
ax[0].set_ylabel("Training loss")
ax[0].set_xlabel("epochs")

ax[0].set_yscale("log")

ax[0].grid()

for k in range(len(hidden_sizes)):
    ax[0].plot(t_epochs, training_losses[k], color=colors[k], alpha=0.7, label="hs = "+str(hidden_sizes[k]))

ax[0].legend()

# Plotting the validation loss results
ax[1].set_title("Final validation loss for different hidden sizes")
ax[1].set_ylabel("Validation loss")
ax[1].set_xlabel("hidden size")

#ax.set_yscale("log")

ax[1].grid()

ax[1].plot(hidden_sizes, [validation_losses[k][-1] for k in range(len(hidden_sizes))], color="k"
        , alpha=0.7, label="loss after 201 epochs", ls=":", marker="d")

ax[1].legend()

fig.savefig("plots/hyperparameter_assessment/hidden_sizes.png")



# Assessment of number of layers

training_losses = []
validation_losses = []

num_layers = [1, 2, 3, 4, 5, 6, 7, 8]
colors = ["red", "orange", "yellow", "green", "blue", "crimson", "black", "grey"]

for nl in num_layers:
    name = "xvpredictor_hyperparameters_assessment_201ep_0_001lr_0_001pp_1ws_32hs_"+str(nl)+"nl_1280ts_512vs_1cm_NONOISE.pth.tar"
    t_epochs, t_loss, v_epochs, v_loss = load_loss(pre_path+name)
    training_losses.append(t_loss)
    validation_losses.append(v_loss)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title("Training loss for different number of layers")
ax[0].set_ylabel("Training loss")
ax[0].set_xlabel("epochs")

ax[0].set_yscale("log")

ax[0].grid()

for k in range(len(num_layers)):
    ax[0].plot(t_epochs, training_losses[k], color=colors[k], alpha=0.7, label="nl = "+str(num_layers[k]))

ax[0].legend()

# Plotting the validation loss results
ax[1].set_title("Final validation loss for different number of layers")
ax[1].set_ylabel("Validation loss")
ax[1].set_xlabel("Number of layers")

#ax.set_yscale("log")

ax[1].grid()

ax[1].plot(num_layers, [validation_losses[k][-1] for k in range(len(num_layers))], color="k"
        , alpha=0.7, label="loss after 201 epochs", ls=":", marker="d")

ax[1].legend()

fig.savefig("plots/hyperparameter_assessment/num_layers.png")



# Assessment of learning rates

training_losses = []
validation_losses = []

learning_rates = [0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
colors = ["red", "orange", "yellow", "green", "blue", "crimson", "black", "grey"]

for lr in learning_rates:
    name = "xvpredictor_hyperparameters_assessment_201ep_"+str(lr).replace(".", "_")+"lr_0_001pp_1ws_32hs_"\
           "5nl_1280ts_512vs_1cm_NONOISE.pth.tar"
    t_epochs, t_loss, v_epochs, v_loss = load_loss(pre_path+name)
    training_losses.append(t_loss)
    validation_losses.append(v_loss)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title("Training loss for different learning rates")
ax[0].set_ylabel("Training loss")
ax[0].set_xlabel("epochs")

ax[0].set_yscale("log")

ax[0].grid()

for k in range(len(learning_rates)):
    ax[0].plot(t_epochs, training_losses[k], color=colors[k], alpha=0.7, label="lr = "+str(learning_rates[k]))

ax[0].legend()

# Plotting the validation loss results
ax[1].set_title("Final validation loss for different learning rates")
ax[1].set_ylabel("Validation loss")
ax[1].set_xlabel("Learning rate")

#ax.set_yscale("log")

ax[1].grid()

ax[1].plot(learning_rates, [validation_losses[k][-1] for k in range(len(learning_rates))], color="k"
        , alpha=0.7, label="loss after 201 epochs", ls=":", marker="d")

ax[1].set_xscale("log")

ax[1].legend()

fig.savefig("plots/hyperparameter_assessment/learning_rates.png")



# Assessment of PINN parameter

training_losses = []
validation_losses = []

pinn_parameter = [1, 0.1, 0.01, 0.001, 0]
colors = ["red", "orange", "yellow", "green", "blue", "crimson", "black", "grey"]

for pp in pinn_parameter:
    name = "xvpredictor_hyperparameters_assessment_201ep_0_001lr_"+str(pp).replace(".","_")+"pp_1ws_32hs_"\
           "5nl_1280ts_512vs_1cm_NONOISE.pth.tar"
    t_epochs, t_loss, v_epochs, v_loss = load_loss(pre_path+name)
    training_losses.append(t_loss)
    validation_losses.append(v_loss)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title("Training loss for different PINN parameters")
ax[0].set_ylabel("Training loss")
ax[0].set_xlabel("epochs")

ax[0].set_yscale("log")

ax[0].grid()

for k in range(len(pinn_parameter)):
    ax[0].plot(t_epochs, training_losses[k], color=colors[k], alpha=0.7, label="p = "+str(pinn_parameter[k]))

ax[0].legend()

# Plotting the validation loss results
ax[1].set_title("Final validation loss for different PINN parameters")
ax[1].set_ylabel("Validation loss")
ax[1].set_xlabel("PINN parameter")

ax[1].set_yscale("log")

ax[1].grid()

ax[1].plot(pinn_parameter, [validation_losses[k][-1] for k in range(len(pinn_parameter))], color="k"
        , alpha=0.7, label="loss after 201 epochs", ls=":", marker="d")

ax[1].hlines(validation_losses[-1][-1], xmin=0.0001, xmax=1, color="grey", alpha=0.7, label="validation loss for p=0")

ax[1].set_xscale("log")

ax[1].legend()

fig.savefig("plots/hyperparameter_assessment/pinn_parameter.png")
