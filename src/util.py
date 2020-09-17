import pandas as pd
import tensorflow as tf
import numpy as np
import os
import re
import math
# plotly?


def generate_rolling_sequence(features, vo2, seq_len, step=5):
    """
    Build a set of sequences (seq_len) from the given features. Use VO2 at the end of the sequence.
    Returns (n, seq_len, nb_features) array
    """
    n = math.ceil((len(vo2)-seq_len)/step)
    nb_features = features.shape[1]  # features is a table, so take #columns
    x = np.zeros((n, seq_len, nb_features))
    y = np.zeros((n,1))
    for i in range(n):
        i0 = i*step
        x[i] = features[i0:i0+seq_len, :]
        y[i] = vo2[i0+seq_len]
    return x, y


def get_x_y_helper(seq_len, feature_list, data_type='train'):
    data_dir = '../data/{}/'.format(data_type)
    csv_files = os.listdir(data_dir)
    # Determine unique participant ids from one of the protocols (high)
    high_matches = [re.search('high(\d+).csv', f) for f in csv_files if f.startswith('high')]
    pids = sorted([int(m.group(1)) for m in high_matches])

    def load_data(pid, protocol):
        # e.g.: high1.csv
        filename = data_dir + '{}{}.csv'.format(protocol, pid)
        df = pd.read_csv(filename)
        # Columns: WR,HR,DeltaHR,VE,BF,VO2rel,HRR,VO2
        vo2 = df['VO2'].to_numpy()
        vo2_rel = df['VO2rel'].to_numpy
        features = df[feature_list].to_numpy()
        x, y = generate_rolling_sequence(features, vo2, seq_len, step=5)
        return x, y

    x = np.array([]).reshape(0, seq_len, len(feature_list))
    y = np.array([]).reshape(0, 1)
    for pid in pids:
        xhi, yhi = load_data(pid, 'high')
        xmid, ymid = load_data(pid, 'mid')
        xlo, ylo = load_data(pid, 'low')

        x = np.concatenate((x, xhi), axis=0)
        y = np.concatenate((y, yhi), axis=0)
        x = np.concatenate((x, xhi, xmid, xlo), axis=0)
        y = np.concatenate((y, yhi, ymid, ylo))
    return x, y


def get_x_y(seq_len=10, feature_list=['HR', 'VE', 'BF', 'HRR']):
    # https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816
    # read all vo2, generate xtrain,ytrain,xtest,ytest
    # note: don't use overlapping participants
    # vo2_data = df['vo2'].values

    # I think (x,y) should be (N,L,F) and (N,1), N=#sequences, L=sequence length, F=#feature channels

    # https://www.tensorflow.org/guide/data
    # If all of your input data fits in memory, the simplest way to create a Dataset from them is to convert them to tf.Tensor objects and use Dataset.from_tensor_slices().

    xtrain, ytrain = get_x_y_helper(seq_len, feature_list, data_type='train')
    xtest, ytest = get_x_y_helper(seq_len, feature_list, data_type='test')
    return xtrain, ytrain, xtest, ytest


    # dataset = tf.data.Dataset.from_tensor_slices((df.to_numpy(), vo2.to_numpy()))
    # train_dataset = dataset.shuffle(len(df)).batch(1)



    # # HR, HRR, RR, VE, WR, all 1hz interpolated
    # filename = '../data/train/whereami.csv'
    # df = pd.read_csv(filename)
    # vo2 = df.pop('vo2').to_numpy()
    # df.pop('time')
    # features = df.to_numpy()
    # seq_len = 10
    # x = np.array([features[i:i+seq_len] for i in range(len(vo2)-seq_len)])
    # y = vo2

    # dataset = tf.data.Dataset.from_tensor_slices((df.to_numpy(), vo2.to_numpy()))
    # train_dataset = dataset.shuffle(len(df)).batch(1)
    # df.head

    #xtrain = np.append(xtrain, vals, axis=0)




    # tf.data.Dataset.from_tensor_slices((sensors, vo2))

    # normalize