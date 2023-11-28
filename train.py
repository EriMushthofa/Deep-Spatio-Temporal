#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
from numpy.core.fromnumeric import argsort
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from src.trainer import Trainer
from src.datamgr import DataMgr, NRELDataMgr, wpDataset
from src.model import Seq2Seq
from src.utils import count_parameters, init_weights

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100
    return mape

def calculate_mase(y_true, y_pred, y_train):
    naive_forecast = y_train[1:]
    mae_naive = np.mean(np.abs(y_train[:-1] - naive_forecast))
    mae_model = np.mean(np.abs(y_true - y_pred))
    mase = mae_model / mae_naive
    return mase


def main():
    parser = argparse.ArgumentParser(
    description="Deep Spatio Temporal Wind Forecasting")
    parser.add_argument("--name", default='wind_power',
                        type=str, help="model name")
    parser.add_argument("--epoch", default=300, type=int, help="max epochs")
    parser.add_argument("--batch_size", default=20000,
                        type=int, help="batch size")
    parser.add_argument("--lr", default=0.001,
                        type=float, help="learning rate")
    parser.add_argument("--k", default=5, type=int,
                        help="number of spatio neighbors")
    parser.add_argument("--n_turbines", default=200, type=int,
                        help="number of turbines")

    args = parser.parse_args()

    print("Running with following command line arguments: {}".
        format(args))

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epoch
    LR = args.lr
    K = args.k
    SAVE_FILE = args.name
    
    if args.name == 'wind_power':
        if not os.path.exists('./data/wind_power.csv'):
            sys.exit('No data found!!!\n'
            'Download wind power data first (follow the instructions in readme).')
        data_mgr = DataMgr(file_path="./data/"+args.name+".csv", K=K)
    elif args.name == 'wind_speed':
        if not os.path.exists('./data/wind_speed.csv'):
            sys.exit('No data found!!!\n'
            'Download wind speed data first (follow the instructions in readme).')
        data_mgr = NRELDataMgr(folder_path="./data/", file_path="wind_speed.csv",
                               meta_path='wind_speed_meta.csv')
    model = Seq2Seq(K=K, n_turbines=args.n_turbines)

    # Cross-validation setup
    k_folds = 3
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_train_loss, all_val_loss = [], []
    all_train_mae, all_val_mae = [], []
    all_train_rmse, all_val_rmse = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(data_mgr.data)))):
        print(f"Training on fold {fold+1}/{k_folds}")

        # Create datasets for this fold
        train_dataset = wpDataset(data_mgr.data[train_idx], ENC_LEN=48, DEC_LEN=12)
        val_dataset = wpDataset(data_mgr.data[val_idx], ENC_LEN=48, DEC_LEN=12)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Initialize model for this fold
        model = Seq2Seq(K=K, n_turbines=args.n_turbines)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        model.apply(init_weights)

        # Trainer for this fold
        trainer = Trainer(model, data_mgr, optimizer, criterion, SAVE_FILE, BATCH_SIZE, ENC_LEN=48, DEC_LEN=12, name=args.name)
        trainer.train(epochs=EPOCHS)

        # Store results
        train_loss, train_mae, train_rmse = trainer.fit()
        val_loss, val_mae, val_rmse = trainer.validate()

        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        all_train_mae.append(train_mae)
        all_val_mae.append(val_mae)
        all_train_rmse.append(train_rmse)
        all_val_rmse.append(val_rmse)

    # Calculate and print average results across all folds
    avg_train_loss = np.mean(all_train_loss)
    avg_val_loss = np.mean(all_val_loss)

    avg_train_mae = np.mean(all_train_mae, axis=0)  # Averaging over folds
    avg_val_mae = np.mean(all_val_mae, axis=0)
    
    avg_train_rmse = np.mean(all_train_rmse, axis=0)
    avg_val_rmse = np.mean(all_val_rmse, axis=0)


    # Print the average results
    print(f'Average Train Loss: {avg_train_loss}')
    print(f'Average Validation Loss: {avg_val_loss}')
    print(f'Average Train MAE: {avg_train_mae}')
    print(f'Average Validation MAE: {avg_val_mae}')
    print(f'Average Train RMSE: {avg_train_rmse}')
    print(f'Average Validation RMSE: {avg_val_rmse}')

    # Save the cross-validation results if needed
    with open('outputs/' + SAVE_FILE + '_cross_val_metrics.txt', 'w') as f:
        f.write(f'Average Train Loss: {avg_train_loss}\n')
        f.write(f'Average Validation Loss: {avg_val_loss}\n')
        f.write(f'Average Train MAE: {avg_train_mae}\n')
        f.write(f'Average Validation MAE: {avg_val_mae}\n')
        f.write(f'Average Train RMSE: {avg_train_rmse}\n')
        f.write(f'Average Validation RMSE: {avg_val_rmse}\n')




    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f'model parameters:{count_parameters(model)}')
    model.apply(init_weights)
    trainer = Trainer(model=model, data_mgr=data_mgr, optimizer=optimizer, criterion=criterion,
                      SAVE_FILE=SAVE_FILE, BATCH_SIZE=BATCH_SIZE, name=args.name)
    trainer.train(epochs=EPOCHS)
    
    y_train = [data[2][:, 1:, 0].numpy() for data in trainer.train_dataloader]
    y_train_np = np.concatenate(y_train)
    loss, mae, rmse, avg_percentage_diff, y_true_concat, y_pred_concat = trainer.report_test_error()
    y_true_np = y_true_concat.numpy()
    y_pred_np = y_pred_concat.numpy()


    mape = calculate_mape(y_true_np, y_pred_np)
    mase = calculate_mase(y_true_np, y_pred_np, y_train_np)

    # Saving additional metrics
    with open('outputs/' + SAVE_FILE + '_metrics.txt', 'w') as f:
        f.write(f'Test Loss: {loss}\n')
        f.write(f'Test MAE: {mae}\n')
        f.write(f'Test RMSE: {rmse}\n')
        f.write(f'Average Percentage Difference: {avg_percentage_diff}%\n')
    
    with open('outputs/' + SAVE_FILE + '_MAPEMASE_metrics.txt', 'w') as f:
        f.write(f'MAPE: {mape}%\n')
        f.write(f'MASE: {mase}\n')


if __name__ == '__main__':
    main()
    print('Done!')
