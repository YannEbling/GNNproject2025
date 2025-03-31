# GNNproject2025

This is the official git repository for the project "A Physics informed neural network for modelling
planetary motion in a two-body system" by Yann Ebling for the course Generative Neural Networks for
Science, submitted on march 31st 2025. No changes will be made after the deadline of 16:00.

All that is found in the report (findings, plots, predictions etc.) can be found in this repository,
along with a lot of older code, which was not used for the final report. The important files are

1. simulation3.py, which is used to create the datasets by solving the equation of motion
2. xv_network.py, which contains the model architecture and the training method
3. network_training.py which governs the network training
4. network_prediction.py, which governs the network inference
5. network_train_and_predict.py, which combines both prior to assess the networks ability to predict
   directly after training

The oudated files are
1. simulation_test.py and simulation2.py, earlier approaches to create simulated datasets, which did
   not work
2. parameters.py, a file, which stored natural constant's for when the simulation was in SI-units
3. auxiliary_functions.py, a collection of functions used for the earlier simulations


The directory models contains all the models that were trained in the process of working on this project.
All of them suffer from the problem described in the report.

The directory plots contains all plots used for the report and many more, which were automatedly created
during model training. The model name is included in the image name and contains information about all
relevant hyperparameters. The number prior to the keyword indicates the numerical value.

The hyperparameter keywords are:

ep - number of epochs, default value 201
lr - learning rate, default value 0.001
pp - PINN-parameter, default value 0.001
ws - window size, only valid option is 1, other window sizes will cause crashes
hs - hidden size, default value 32
nl - number of layers, default value 5
ts - training set size, default value 1280
vs - validation set size, default value 512
cm - "conservation mode", used to distinguish between whether energy conservation (cm="1"), 
     conservation of angular momentum ("2") or both ("12") are included in the loss. Only
     energy conservation is used, so conservation mode "2" or empty "" are valid options

This leads to the filename 
xvpredictor_hyperparameters_assessment_201ep_0_001lr_0_001pp_1ws_32hs_5nl_1280ts_512vs_1cm_NONOISE.pth.tar
for the model, which can be found in the model directory.
