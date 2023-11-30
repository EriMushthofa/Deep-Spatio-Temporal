#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.trainer import Trainer
from src.datamgr import DataMgr, NRELDataMgr, wpDataset
from src.model import Seq2Seq
from src.utils import count_parameters, init_weights

def calculate_mape(y_true, y_pred):
    # Calculate Mean Absolute Percentage Error
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100
    return mape

def calculate_mase(y_true, y_pred, y_train):
    # Calculate Mean Absolute Scaled Error
    naive_forecast = y_train[1:]
    mae_naive = np.mean(np.abs(y_train[:-1] - naive_forecast))
    mae_model = np.mean(np.abs(y_true - y_pred))
    mase = mae_model / mae_naive
    return mase

def plot_forecast(y_true, y_pred, title='Forecast vs Actuals'):
    # Plotting the forecast against actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Forecast')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Deep Spatio Temporal Wind Forecasting")
    parser.add_argument("--name", default='wind_power', type=str, help="model name")
    parser.add_argument("--epoch", default=300, type=int, help="max epochs")
    parser.add_argument("--batch_size", default=20000, type=int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--k", default=5, type=int, help="number of spatio neighbors")
    parser.add_argument("--n_turbines", default=200, type=int, help="number of turbines")
    args = parser.parse_args()

    # Model and data setup
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epoch
    LR = args.lr
    K = args.k
    SAVE_FILE = args.name

    if args.name == 'wind_power':
        if not os.path.exists('./data/wind_power.csv'):
            sys.exit('No data found!!!\nDownload wind power data first.')
        data_mgr = DataMgr(file_path="./data/"+args.name+".csv", K=K)
    elif args.name == 'wind_speed':
        if not os.path.exists('./data/wind_speed.csv'):
            sys.exit('No data found!!!\nDownload wind speed data first.')
        data_mgr = NRELDataMgr(folder_path="./data/", file_path="wind_speed.csv", meta_path='wind_speed_meta.csv')

    # Cross-validation setup
    k_folds = 3
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Storing metrics for each fold
    all_metrics = {'train_mae': [], 'val_mae': [], 'test_mae': [], 
                   'train_rmse': [], 'val_rmse': [], 'test_rmse': [], 
                   'mape': [], 'mase': []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(data_mgr.data)))):
        if fold >= 3:
            break  # Stop after the first 3 folds

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

        # Training and validation
        trainer = Trainer(model, data_mgr, optimizer, criterion, SAVE_FILE, BATCH_SIZE, ENC_LEN=48, DEC_LEN=12, name=args.name)
        trainer.train(epochs=EPOCHS)
        train_loss, train_mae, train_rmse = trainer.fit()
        val_loss, val_mae, val_rmse = trainer.validate()
        test_lost, test_mae, test_rmse, average_percentage_diff, y_true_concat, y_pred_concat = trainer.report_test_error()


        # Evaluate the model on the test set
        y_train = [data[2][:, 1:, 0].numpy() for data in trainer.train_dataloader]
        y_train_np = np.concatenate(y_train)
        loss, mae, rmse, avg_percentage_diff, y_true_concat, y_pred_concat = trainer.report_test_error()
        y_true_np = y_true_concat.numpy()
        y_pred_np = y_pred_concat.numpy()

        

        # Store metrics for this fold
        all_metrics['train_mae'].append(np.mean(train_mae))
        all_metrics['val_mae'].append(np.mean(val_mae))
        all_metrics['test_mae'].append(np.mean(mae))
        all_metrics['train_rmse'].append(np.mean(train_rmse))
        all_metrics['val_rmse'].append(np.mean(val_rmse))
        all_metrics['test_rmse'].append(np.mean(rmse))
        

        # Plotting the forecast
        plot_forecast(y_true_np[0], y_pred_np[0], title=f'Fold {fold+1} Forecast vs Actuals')

    # Save the cross-validation results
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv('outputs/New_Output.csv', index=False)
    print('Cross-validation results saved to New_Output.csv')

if __name__ == '__main__':
    main()
    print('Done!')

