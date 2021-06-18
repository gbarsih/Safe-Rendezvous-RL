import torch
import torch.nn as nn
import numpy as np
import seaborn as sns;

import osmnx as ox
import networkx as nx
import importlib as im
import utils
import time
import pickle
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import itertools
import csv


im.reload(utils)

figsize = utils.figsize
dpi = utils.dpi
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

city_list = ['champaign','dtchicago','chicago','janeiro']
n_l_list = [2,3]
n_n_list = [32, 128, 512]
d_p_list = [0.2, 0.4, 0.8]
activation_list = ['SELU', 'ReLU', 'Sigmoid']
all_list = [city_list, n_l_list, n_n_list, d_p_list, activation_list]
res = list(itertools.product(*all_list))

class ModelSELU(nn.Module):

    def __init__(self, input_size, output_size, layers, p=0.2):
        super().__init__()

        all_layers = []

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.SELU())  # Leaky Doing well
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x

class ModelReLU(nn.Module):

    def __init__(self, input_size, output_size, layers, p=0.2):
        super().__init__()

        all_layers = []

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.SELU())  # Leaky Doing well
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x

class ModelSigmoid(nn.Module):

    def __init__(self, input_size, output_size, layers, p=0.2):
        super().__init__()

        all_layers = []

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.Sigmoid())  # Leaky Doing well
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x


with open('log_perf.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Location", "n_l", "n_n", "p_d", "MSE", "MAE", "Loss STD"])
    for leg in res:
        print(leg)
        city = leg[0]
        n_l = leg[1]
        n_n = leg[2]
        d_p = leg[3]
        act = leg[4]
        G = utils.getGraphWithSetting(city)
        risks, deltas, routes = utils.getDatasetFromCity(city)

        oxp, oyp, dxp, dyp, rrv, rxp, ryp, ddv, coord_bounds, depot_local, depot_node = utils.unPackDataset(city, G, risks,
                                                                                                        deltas, routes)

        ur = coord_bounds['ur']

        df = pd.DataFrame(list(zip(oxp, oyp, dxp, dyp, rrv, ddv)),
                          columns=['X_origin', 'Y_origin', 'X_destination', 'Y_destination', 'risks', 'deltas'])
        Inputs = ['X_origin', 'Y_origin', 'X_destination', 'Y_destination']
        Outputs = ['deltas']

        xo = df['X_origin'].values
        yo = df['Y_origin'].values
        xd = df['X_destination'].values
        yd = df['Y_destination'].values
        r = df['deltas'].values

        loss_arr = []
        mse_arr = []
        mae_arr = []
        nruns = 10
        for j in range(nruns):
            print('run',j+1,'of',nruns)

            inp_stack = torch.tensor(df[Inputs].values, dtype=torch.float64)
            out_stack = torch.tensor(df[Outputs].values, dtype=torch.float32)

            total_data_entries = len(r)
            test_data = int(total_data_entries * .1)

            train_inputs = inp_stack[:total_data_entries - test_data].to(device=device)
            train_outputs = out_stack[:total_data_entries - test_data].to(device=device)

            test_inputs = inp_stack[total_data_entries - test_data:total_data_entries].to(device=device)
            test_outputs = out_stack[total_data_entries - test_data:total_data_entries].to(device=device)

            rxpt = rxp[total_data_entries - test_data:total_data_entries]
            rypt = ryp[total_data_entries - test_data:total_data_entries]

            lr = 0.001

            layers = []
            for i in range(n_l):
                layers.append(n_n)  # 32-128 range with low dropout (0.1-0.4) seems to work the best

            if act == 'SELU':
                model = ModelSELU(4, 1, layers, p=d_p)
            elif act == 'ReLU':
                model = ModelReLU(4, 1, layers, p=d_p)
            elif act == 'Sigmoid':
                model = ModelSigmoid(4, 1, layers, p=d_p)

            model = nn.DataParallel(model)
            model.to(device)
            epochs = 20000
            aggregated_losses = []

            loss_function = nn.MSELoss().to(device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            model.train()
            start = time.time()
            for i in range(epochs):
                i += 1
                y_pred = model(train_inputs.float())
                single_loss = loss_function(y_pred, train_outputs)
                aggregated_losses.append(single_loss)

                if i % 25 == 1:
                    end = time.time()
                    elapsed = end - start
                    print(f'epoch: {i:3} loss: {single_loss.item():10.8f} time: {elapsed:10.3f}')
                    start = time.time()

                optimizer.zero_grad()
                single_loss.backward()
                optimizer.step()

            print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

            with torch.no_grad():
                model.eval()
                predictions = model(test_inputs.float())

            predictions = predictions.cpu().numpy()
            reals = test_outputs.cpu().numpy()

            test_size = len(reals)

            for i in range(10):
                pred = predictions[i]

            idxs = [i for i in range(len(reals)) if reals[i] > 0]
            rrvp = [reals[i] for i in idxs]
            rxpp = [rxpt[i] for i in idxs]
            rypp = [rypt[i] for i in idxs]

            preds = [np.abs(predictions[i][0]) for i in range(len(rrvp))]
            perf = [np.abs(predictions[i][0] - rrvp[i]) for i in range(len(rrvp))]
            perf_pct = [predictions[i][0] / rrvp[i] for i in range(len(rrvp))]
            perf_mse = [(predictions[i][0] - rrvp[i]) ** 2 for i in range(len(rrvp))]
            perf_mae = [np.abs(predictions[i][0] - rrvp[i]) for i in range(len(rrvp))]

            pu = np.max(perf)
            pl = np.min(perf)

            perfp = [utils.mapRange(x, pl, pu, lr, ur) for x in perf]
            perf_mse = np.mean(perf_mse)
            perf_mae = np.mean(perf_mae)
            print('Median', np.median(perf))
            print('MSE', perf_mse)
            print('MAE', perf_mae)
            print('Max Err', np.max(np.abs(perf)))

            loss_arr.append(single_loss.item())
            mse_arr.append(perf_mse)
            mae_arr.append(perf_mae)

        # writer.writerow(["Location", "$n_l$", "n_n", "$p_d$", "MSE", "MAE", "Loss STD"])

        del inp_stack
        del out_stack
        del train_inputs
        del train_outputs
        del test_inputs
        del test_outputs
        del rxpt
        del rypt
        del loss_function
        del model
        del predictions

        torch.cuda.empty_cache()

        loss_std = "{:.4e}".format(np.std(loss_arr))
        mmse = "{:.4e}".format(np.mean(mse_arr))
        mmae = "{:.4e}".format(np.mean(mae_arr))

        fields = [utils.getCityName(city), n_l, n_n, d_p, mmse, mmae, loss_std]
        print(fields)
        writer.writerow(fields)
