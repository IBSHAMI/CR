import time
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import random
from collections import deque
import torch


#######################################################################################################
############################ Data preparation section #################################################


def data_preparation(path, slice_ind, columns_drop_list, EMA_num=3, EMA_periods_list=[3, 6, 9],
                     SMA_period=20,
                     future_price_predict=3):
    # read the excel file
    df = pd.read_excel(path, header=None, skiprows=1)

    # drop the header, replace with first row and drop first row 
    df.columns = df.iloc[0]
    df = df.iloc[1:, :]

    # reverse the dataframe and reset the index numbering
    df = df.iloc[::-1]
    # slice the index to the latest data
    df = df.iloc[slice_ind:]
    df = df.reset_index(drop=True)

    df_length = len(df)  # number of rows in the dataframe

    # add hour and minute columns
    # df["hour"] = pd.to_datetime(df["date"]).dt.hour
    # df["minute"] = pd.to_datetime(df["date"]).dt.minute
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.strftime('%m/%d/%Y')

    mid_price_1 = (df["high"] + df["low"]) / 2
    mid_price_2 = (df["open"] + df["close"]) / 2
    df.drop(columns=columns_drop_list, inplace=True)

    df["mid_price_1"] = mid_price_1
    df["mid_price_2"] = mid_price_2

    # --------------- EMA calculation -----------------------#
    def calculate_ema(prices, period, smoothing=2):
        ema = [sum(prices[:period]) / period]
        for price in prices[period:]:
            ema.append((price * (smoothing / (1 + period))) + ema[-1] * (1 - (smoothing / (1 + period))))
        return ema

    EMA_list = []
    if EMA_num == len(EMA_periods_list):
        str_names_list = []
        for i in range(EMA_num):
            str_names_list.append(F"EMA{i}")

        for period in EMA_periods_list:
            EMA_list.append(pd.DataFrame(calculate_ema(df['close'], period)))

    # slice the df and the EMA dataframes to combine
    slice_index = EMA_periods_list[-1] + 1
    df = df[slice_index:]
    for index in range(len(EMA_list)):
        EMA_list[index] = EMA_list[index][slice_index - (df_length - len(EMA_list[index])):]

    for name, data in zip(str_names_list, EMA_list):
        df[name] = data

    # ---------------- RSI calculation ------------------------#
    # difference between each 2 lines                                                    
    delta = df["close"].diff(1)
    delta.dropna(inplace=True)

    positive = delta.copy()
    negative = delta.copy()
    positive[positive < 0] = 0
    negative[negative > 0] = 0

    mins = 14  # change to minutes
    average_gain = positive.rolling(window=mins).mean()
    average_loss = abs(negative.rolling(window=mins).mean())

    relative_strength = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))

    # add the RSI data to the dataframe
    df["RSI"] = RSI
    df.dropna(inplace=True)

    # ------------------------------ Bollinger Bands and SMA --------------------------#
    # calculate the simple moving average for 20 min period
    df["SMA"] = df["close"].rolling(window=SMA_period).mean()

    # calculate starndard deviation 
    STD = df["close"].rolling(window=SMA_period).std()

    # upper limit 
    df["upper"] = df["SMA"] + (STD * 2)
    # lower limit                                                     
    df["lower"] = df["SMA"] - (STD * 2)

    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    # ------------------------- Tragets colmuns ---------------------------------#
    def future_price_change(current, future):
        if float(future) > float(current):
            return 1
        else:
            return 0

    df["future_price"] = df["close"].shift(-future_price_predict)
    df["future_change"] = list(map(future_price_change, df["close"], df["future_price"]))
    df.dropna(inplace=True)

    return df


def plot_data(df):
    plt.figure(figsize=(20, 18))
    ax1 = plt.subplot(211)
    ax1.plot(df.index, df["close"], label="Close_price", color="lightgray")
    ax1.plot(df.index, df['EMA0'], label="EMA0")
    ax1.plot(df.index, df['EMA1'], label="EMA1")
    ax1.plot(df.index, df['EMA2'], label="EMA2")
    ax1.fill_between(df.index, df['upper'], df['lower'], color="gold")
    ax1.set_title("close price", color="#ffffff", fontsize=20)

    # ax1.title("Close Price", color="white")
    ax1.grid(True, color="#555555")
    ax1.set_axisbelow(True)
    ax1.set_facecolor("black")
    ax1.figure.set_facecolor("#121212")
    ax1.tick_params(axis='y', colors='#FFFFFF')
    ax1.tick_params(axis='x', colors='#FFFFFF')
    ax1.set_xlabel('Date', color="#ffffff", fontsize=16)
    ax1.set_ylabel('Price', color="#ffffff", fontsize=16)

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(df.index, df["RSI"], color="lightgray")
    ax2.axhline(0, linestyle="--", alpha=0.5, color="#ff0000")
    ax2.axhline(10, linestyle="--", alpha=0.5, color="#ffaa00")
    ax2.axhline(20, linestyle="--", alpha=0.5, color="#00ff00")
    ax2.axhline(30, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.axhline(70, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.axhline(80, linestyle="--", alpha=0.5, color="#00ff00")
    ax2.axhline(90, linestyle="--", alpha=0.5, color="#ffaa00")
    ax2.axhline(100, linestyle="--", alpha=0.5, color="#ff0000")

    # ax2.title("RSI", color="white")
    ax2.set_title("RSI", color="#ffffff", fontsize=20)
    ax2.grid(False)
    ax2.set_axisbelow(True)
    ax2.set_facecolor("black")
    ax2.tick_params(axis='y', colors='#FFFFFF')
    ax2.tick_params(axis='x', colors='#FFFFFF')

    plt.show()


#######################################################################################################
#######################################################################################################


#######################################################################################################
############################ Data preprocessing section ###############################################


def create_bollinger_limits(df):
    upper_limit, lower_limit = [], []
    for upper, lower, close in zip(df["upper"], df["lower"], df["close"]):
        avg_bollinger = (upper + lower) / 2
        if close >= avg_bollinger:
            upper_limit.append(1)
            lower_limit.append(0)
        elif close < avg_bollinger:
            upper_limit.append(0)
            lower_limit.append(1)
        else:
            print("error")

    return upper_limit, lower_limit


def pct_change(df, columns_avoid):
    for col in df.columns:
        if col not in columns_avoid:
            df[col] = pd.to_numeric(df[col])
            df[col] = df[col].pct_change()
            q_low = df[col].quantile(0.015)
            q_hi = df[col].quantile(0.985)
            df = df[(df[col] < q_hi) & (df[col] > q_low)]
            df[col] = df[col] * 1000
    q_low = df["Volume USDT"].quantile(0.015)
    q_hi = df["Volume USDT"].quantile(0.985)
    df = df[(df["Volume USDT"] < q_hi) & (df["Volume USDT"] > q_low)]
    df.dropna(inplace=True)
    # df = df.reset_index(drop=True)
    return df


def preprocess_and_split_df(df, new_column_index, columns_to_scale):
    # reindex the columns
    df = df.reindex(new_column_index, axis=1)
    # Normalize using Min and Max
    ind_times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1 * len(ind_times))]  # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2 * len(ind_times))]  # Last 20% of series

    for col in columns_to_scale:
        # find the mean
        mean = df[col].mean()
        # find the std
        std = df[col].std()
        # scale the data
        df[col] = (df[col] - mean) / std

    # split the data
    df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    df_test = df[(df.index >= last_10pct)]

    return df, df_train, df_val, df_test


#######################################################################################################
############################ Datasets and Modeling ###############################################

def create_datasets(array, columns_length, seq_len=60):
    # prepare data chunks for training
    x, y = [], []

    # training data
    for i in range(seq_len, len(array)):
        if i == len(array):
            break
        else:
            x.append(array[:, :columns_length][i - seq_len:i])
            y.append(array[:, columns_length][i])

    return x, y


def convert_to_dataset(df, data_columns, columns_length, seq_len=60):
    data_array = df[data_columns].values
    x, y = create_datasets(data_array, columns_length, seq_len)
    y = [np.array([element]) for element in y]
    dataset = []
    for x, y in zip(x, y):
        dataset.append([torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))])
    return dataset

#######################################################################################################
#######################################################################################################
