import pandas as pd
import numpy as np
import os
import re
import math
import csv


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


def _get_x_y_helper(seq_len, feature_list, seq_step=5, data_type='train', vo2_type='VO2'):
    data_dir = '../data/{}/'.format(data_type)
    csv_files = os.listdir(data_dir)
    # Determine full set of unique participant ids
    high_matches = [re.search('(\d+).csv', f) for f in csv_files if f.startswith('high')]
    pids = sorted([int(m.group(1)) for m in high_matches])
    pids = list(set(pids))  # unique ids

    # TODO: use either filenames or pids and get rid of the other.

    def load_data_pid(pid, protocol):
        # e.g.: high1.csv
        filename = '{}{}.csv'.format(protocol, pid)
        try:
            df = pd.read_csv(data_dir + filename)
        except FileNotFoundError:
            return None, None

        # Columns: WR,HR,DeltaHR,VE,BF,VO2rel,HRR,VO2
        vo2 = df[vo2_type].to_numpy()
        features = df[feature_list].to_numpy()
        x_, y_ = generate_rolling_sequence(features, vo2, seq_len, step=seq_step)
        return x_, y_

    def load_data(filename):
        # e.g.: high1.csv
        # filename = data_dir + '{}{}.csv'.format(protocol, pid)
        try:
            df = pd.read_csv(data_dir + filename)
        except FileNotFoundError:
            return None, None

        # Columns: WR,HR,DeltaHR,VE,BF,VO2rel,HRR,VO2
        vo2 = df[vo2_type].to_numpy()
        features = df[feature_list].to_numpy()
        x_, y_ = generate_rolling_sequence(features, vo2, seq_len, step=seq_step)
        return x_, y_

    x = np.array([]).reshape(0, seq_len, len(feature_list))
    y = np.array([]).reshape(0, 1)
    xstatic = np.array([]).reshape(0, 3)
    # for pid in pids[:2]:
    df = pd.read_csv(data_dir + 'demographics.csv')
    demogs = {p: [h, w, a] for p, h, w, a in zip(df['Participant'], df['Height'], df['Weight'], df['Age'])}
    for f in csv_files:
        if f == 'demographics.csv':
            continue
        xi, yi = load_data(f)

        match = re.search('(\d+).csv', f)
        pid = int(match.group(1))
        xstatic_i = np.array(demogs[pid]).reshape(1, 3)
        # Repeat these demographics for all of this participant's data
        xstatic_i = np.tile(xstatic_i, (xi.shape[0], 1))

        #print('{}: {}-{}'.format(f, x.shape[0], x.shape[0]+xi.shape[0]))
        x = np.concatenate((x, xi), axis=0)
        y = np.concatenate((y, yi))

        xstatic = np.concatenate((xstatic, xstatic_i), axis=0)

        # TODO: uncomment low and mid
        # xhi, yhi = load_data(pid, 'high')
        # xmid, ymid = load_data(pid, 'mid')
        # xlo, ylo = load_data(pid, 'low')
        # x = np.concatenate((x, xhi, xmid, xlo), axis=0)
        # y = np.concatenate((y, yhi, ymid, ylo))
    return x, xstatic, y


def get_x_y(seq_len=10, feature_list=['HR', 'VE', 'BF', 'HRR'], seq_step_train=5, vo2_type='VO2'):
    """
    Generates train and test sequences.
    :param seq_len: sequence length (s)
    :param feature_list: headers from .csv files
    :param seq_step_train: step between sequences during training
    :param vo2_type: VO2 or VO2rel
    :return: xtrain, ytrain, xval, yval, xtest, ytest where x [n,seq_len,nb_feat], y [n,1]
    """
    # https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816
    # read all vo2, generate xtrain,ytrain,xtest,ytest
    # note: don't use overlapping participants
    # vo2_data = df['vo2'].values

    # I think (x,y) should be (N,L,F) and (N,1), N=#sequences, L=sequence length, F=#feature channels

    # https://www.tensorflow.org/guide/data
    # https://www.tensorflow.org/tutorials/structured_data/time_series
    # https://www.tensorflow.org/tutorials/load_data/csv
    # If all of your input data fits in memory, the simplest way to create a Dataset from them is to convert them to tf.Tensor objects and use Dataset.from_tensor_slices().

    xtrain, xtrain_static, ytrain = _get_x_y_helper(seq_len, feature_list, seq_step=seq_step_train, data_type='train', vo2_type=vo2_type)
    # Get every sequence in the data for testing
    xval, xval_static, yval = _get_x_y_helper(seq_len, feature_list, seq_step=1, data_type='validation', vo2_type=vo2_type)
    xtest, xtest_static, ytest = _get_x_y_helper(seq_len, feature_list, seq_step=1, data_type='test', vo2_type=vo2_type)

    # TODO: change to estimators? https://towardsdatascience.com/how-to-normalize-features-in-tensorflow-5b7b0e3a4177
    mu = np.mean(xtrain, axis=0)
    sigma = np.std(xtrain, axis=0)
    xtrain = (xtrain-mu)/sigma
    xval = (xval-mu)/sigma
    xtest = (xtest-mu)/sigma

    return xtrain, xtrain_static, ytrain, xval, xval_static, yval, xtest, xtest_static, ytest


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