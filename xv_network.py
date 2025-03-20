import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class XVPredictor(nn.Module):

    def __init__(self, window_size=10, input_size=5, output_size=2, hidden_size=32, num_layers=5, G_M=4*torch.pi**2,
                 *args, **kwargs):

        super(XVPredictor, self).__init__()

        self.window_size = window_size
        self.input_size = input_size
        self.G_M = G_M

        # initialization of the layer list, with the input layer + activation
        super().__init__(*args, **kwargs)
        linear_layer_list = [nn.Linear(1 + window_size*input_size, hidden_size)]
        activation_layer_list = [nn.Tanh()]

        for _ in range(num_layers):
            # adding each hidden layer with a linear layer and a tanh activation layer
            linear_layer_list.append(nn.Linear(hidden_size, hidden_size))
            activation_layer_list.append(nn.Tanh())

        linear_layer_list.append(nn.Linear(hidden_size, output_size))

        self.linear_layers = nn.ModuleList(linear_layer_list)
        self.activation_layers = nn.ModuleList(activation_layer_list)

    def forward(self, x_in, der=False):
        x = 0
        if not der:
            x = x_in
            for k in range(len(self.activation_layers)):
                x = self.linear_layers[k](x)
                x = self.activation_layers[k](x)
            x = self.linear_layers[-1](x)

        if der:
            y = x_in[:3]
            v = x_in[3:]
            a = -self.G_M * y / torch.norm(y, p=2)**3

            x = torch.cat((v, a))
            for k in range(len(self.activation_layers)):
                x = torch.mv(self.linear_layers[k].weight, x)
                x = 1 - self.activation_layers[k](x)**2
            x = torch.mv(self.linear_layers[-1].weight, x)

        return x

    def predict_trajectory(self, timepoints, x_in):
        results = torch.zeros(size=(timepoints.shape[0], self.input_size-1))
        x = torch.zeros(1 + self.window_size * self.input_size)
        for j in range(x_in.shape[0]):
            x[-j] = x_in[-j]

        for t_ind in range(timepoints.shape[0]):
            t = timepoints[t_ind]
            x[0] = t
            x_pred = self.forward(x, der=False)
            v_pred = self.forward(x, der=True)

            x_new = torch.cat((x_pred, v_pred))

            results[t_ind] = x_new
            updated_x = torch.zeros_like(x)
            updated_x[1:-4] = x[5:]
            updated_x[-4:] = x_new

            x = updated_x
            assert x[0] == 0

        return results


def energy_loss(x_pred, E, G_M):

    x = x_pred[:2]
    v = x_pred[2:]

    v_norm = torch.norm(v, p=2.)
    x_norm = torch.norm(x, p=2.)

    return torch.abs(v_norm**2 / 2 - G_M / x_norm - E)


def angular_momentum_loss(x_pred, h):
    x = x_pred[:2]
    v = x_pred[2:]

    x_norm = torch.norm(x, p=2)
    v_norm = torch.norm(v, p=2)

    gamma = torch.acos(torch.dot(x, v) / (x_norm * v_norm))

    return torch.abs(x_norm * v_norm * torch.sin(gamma) - h)


def train_xv_predictor(training_set, validation_set, model, hyperparameters, conservation_mode="1",
                       G_M = 4*torch.pi**2):
    epochs = hyperparameters["epochs"]
    lr = hyperparameters["lr"]
    batch_size = hyperparameters["batch_size"]
    pinn_parameter = hyperparameters["pinn_parameter"]

    N = training_set.shape[0]
    T = training_set.shape[1]
    D = training_set.shape[2]

    N_vali = validation_set.shape[0]
    T_vali = validation_set.shape[1]
    D_vali = validation_set.shape[2]

    window_size = model.window_size

    model.train()

    mse_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    validation_loss = []


    ind = np.arange(N)

    for epoch in range(epochs):
        print("Currently in epoch ", epoch, ".")
        if epoch % 10 == 0:
            print("Training in progress: Currently in epoch ", epoch, ". Progress: ", 100*round(epoch/epochs, 3), "%.")

        epoch_loss = []

        np.random.shuffle(ind)

        optimizer.zero_grad()

        for i in ind:
            orbit_data = training_set[i, :, :]
            orbit_loss = 0

            for k in range(window_size+1, T):
                input_values = orbit_data[k-window_size-1:k-1, :]

                input_values = input_values.ravel()

                t_k = orbit_data[k, 0]
                x_k = orbit_data[k, 1:]

                input_values_and_time = torch.zeros((1 + input_values.shape[0]))
                input_values_and_time[0] = t_k
                input_values_and_time[1:] = input_values

                input_values = input_values_and_time

                y_k_hat = model(input_values, der=False)
                v_k_hat = model(input_values, der=True)

                x_k_hat = torch.cat((y_k_hat, v_k_hat))

                pinn_loss = 0

                for letter in conservation_mode:
                    if letter == "1":
                        v_k_minus1 = orbit_data[k-1, 3:]
                        x_k_minus1 = orbit_data[k-1, 1:3]

                        v_norm = torch.norm(v_k_minus1, p=2)
                        x_norm = torch.norm(x_k_minus1, p=2)

                        E = v_norm**2 / 2 - G_M / x_norm

                        E_loss = energy_loss(x_k_hat, E=E, G_M=G_M)

                        pinn_loss += E_loss

                    if letter == "2":
                        v_k_minus1 = orbit_data[k-1, 3:]
                        x_k_minus1 = orbit_data[k-1, 1:3]

                        v_norm = torch.norm(v_k_minus1, p=2)
                        x_norm = torch.norm(x_k_minus1, p=2)

                        gamma = torch.acos(torch.dot(x_k_minus1, v_k_minus1) / (x_norm * v_norm))

                        h = x_norm * v_norm * torch.sin(gamma)

                        h_loss = angular_momentum_loss(x_k_hat, h=h)

                        pinn_loss += h_loss

                data_loss = mse_loss(x_k_hat, x_k)

                loss = data_loss + pinn_parameter * pinn_loss

                orbit_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss.append(orbit_loss)
        train_loss.append(epoch_loss)


        # ____________________________________________________________________________________________
        # VALIDATION LOSS SECTION BELOW! VALIDATION LOSS SECTION BELOW! VALIDATION LOSS SECTION BELOW!
        #_____________________________________________________________________________________________

        if epoch % 10 == 0:
            epoch_vali_loss = []
            with torch.no_grad():
                for i in range(N_vali):
                    orbit_data = validation_set[i, :, :]
                    orbit_loss = 0

                    for k in range(window_size + 1, T):
                        input_values = orbit_data[k - window_size - 1:k - 1, :]

                        input_values = input_values.ravel()

                        t_k = orbit_data[k, 0]
                        x_k = orbit_data[k, 1:]

                        input_values_and_time = torch.zeros((1 + input_values.shape[0]))
                        input_values_and_time[0] = t_k
                        input_values_and_time[1:] = input_values

                        input_values = input_values_and_time

                        y_k_hat = model(input_values, der=False)
                        v_k_hat = model(input_values, der=True)

                        x_k_hat = torch.cat((y_k_hat, v_k_hat))

                        pinn_loss = 0

                        for letter in conservation_mode:

                            if letter == "1":
                                v_k_minus1 = orbit_data[k - 1, 3:]
                                x_k_minus1 = orbit_data[k - 1, 1:3]

                                v_norm = torch.norm(v_k_minus1, p=2)
                                x_norm = torch.norm(x_k_minus1, p=2)

                                E = v_norm ** 2 / 2 - G_M / x_norm

                                pinn_loss += energy_loss(x_k_hat, E=E, G_M=G_M)

                            if letter == "2":
                                v_k_minus1 = orbit_data[k - 1, 3:]
                                x_k_minus1 = orbit_data[k - 1, 1:3]

                                v_norm = torch.norm(v_k_minus1, p=2)
                                x_norm = torch.norm(x_k_minus1, p=2)

                                gamma = torch.acos(torch.dot(x_k_minus1, v_k_minus1) / (x_norm * v_norm))

                                h = x_norm * v_norm * torch.sin(gamma)

                                pinn_loss += angular_momentum_loss(x_k_hat, h=h)

                        loss = mse_loss(x_k_hat, x_k) + pinn_parameter * pinn_loss

                        orbit_loss += loss.item()
                    epoch_vali_loss.append(orbit_loss)
                validation_loss.append(epoch_loss)

    return train_loss, validation_loss

