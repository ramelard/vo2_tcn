import pandas as pd
import numpy as np
import os
import re
import math
import tensorflow as tf
from tensorflow.keras import Model, Input, optimizers
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras import initializers
from tcn import TCN
import csv


def build_model(opts, use_demographics=False, nb_static=None):
    # "The most important factor for picking parameters is to make sure that
    # the TCN has a sufficiently large receptive field by choosing k and d
    # that can cover the amount of context needed for the task." (Bai 2018)
    # Receptive field = nb_stacks * kernel_size * last_dilation
    # opts keys: max_len, nb_feat, nb_filters, kernel_size, dilations, dropout_rate, lr
    input_layer = Input(shape=(opts['max_len'], opts['nb_feat']))
    x = TCN(opts['nb_filters'], opts['kernel_size'], 1, opts['dilations'], 'causal',
            False, opts['dropout_rate'], False,
            'relu', 'he_normal', False, True,
            name='tcn')(input_layer)

    input_layers = [input_layer]
    if use_demographics:
        # TODO: use Embedding layer with one-hot indexing (see Esteban 2015/2016), or use directly.
        input_layer_static = Input(shape=(nb_static,))
        # y = Embedding(nb_static, nb_static)(input_layer_static)
        x = tf.keras.layers.concatenate([x, input_layer_static])
        input_layers.append(input_layer_static)

    z = Dense(1)(x)
    z = Activation('linear')(z)
    output_layer = z
    model = Model(input_layers, output_layer)
    opt = optimizers.Adam(lr=opts['lr'], clipnorm=1.)
    model.compile(opt, loss='mean_squared_error')
    # print('model.x = {}'.format([l.shape for l in input_layers]))
    # print('model.y = {}'.format(output_layer.shape))
    return model


def build_zignoliLSTM(opts):
    # Stacked LSTM model from Zignoli 2020
    # nb_timesteps = 105 is approx 70 breaths, assuming BF=40
    nb_timesteps = opts['max_len']
    nb_feat = opts['nb_feat']
    inputs = Input(shape=(nb_timesteps, nb_feat))
    x = LSTM(units=32, return_sequences=True,
             bias_initializer=initializers.RandomUniform(minval=0, maxval=0.1))(inputs)
    x = LSTM(units=32, return_sequences=True,
             bias_initializer=initializers.RandomUniform(minval=0, maxval=0.1))(x)
    x = LSTM(units=32,
             bias_initializer=initializers.RandomUniform(minval=0, maxval=0.1))(x)
    x = Dense(10, bias_initializer=initializers.RandomUniform(minval=0, maxval=0.1))(x)
    outputs = Dense(1, bias_initializer=initializers.RandomUniform(minval=0, maxval=0.1))(x)

    model = Model(inputs, outputs)
    opt = optimizers.Adagrad(learning_rate=opts['lr'])
    model.compile(opt, loss='categorical_crossentropy')
    return model


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
    descr = np.array([]).reshape(0, 2)  # pid, protocol
    # for pid in pids[:2]:
    df = pd.read_csv(data_dir + '../demographics.csv')
    demogs = {p: [h, w, a] for p, h, w, a in zip(df['Participant'], df['Height'], df['Weight'], df['Age'])}
    for f in csv_files:
        xi, yi = load_data(f)
        match = re.search('(high|mid|low|max)(\d+)(_\d)*.csv', f)
        if match is None:
            print(f'Could not interpret data file {f}. Skipping.')
            continue
        protocol = match.group(1)
        pid = int(match.group(2))
        xstatic_i = np.array(demogs[pid]).reshape(1, 3)
        # Repeat these demographics for all of this participant's data
        xstatic_i = np.tile(xstatic_i, (xi.shape[0], 1))
        descr_i = np.tile((protocol, pid), (xi.shape[0], 1))

        #print('{}: {}-{}'.format(f, x.shape[0], x.shape[0]+xi.shape[0]))
        x = np.concatenate((x, xi), axis=0)
        y = np.concatenate((y, yi))
        xstatic = np.concatenate((xstatic, xstatic_i), axis=0)
        descr = np.concatenate((descr, descr_i), axis=0)
    return x, xstatic, y, descr


def get_x_y(seq_len=10, feature_list=['WR', 'HR', 'VE', 'BF', 'HRR'], seq_step_train=5, vo2_type='VO2', norm_type='mixed'):
    """
    Generates train and test sequences.
    :param seq_len: sequence length (s)
    :param feature_list: headers from .csv files
    :param seq_step_train: step between sequences during training
    :param vo2_type: VO2 or VO2rel
    :param norm_type: mixed (WR norm, standardize other), norm, or stand
    :return: xtrain, ytrain, xval, yval, xtest, ytest where x [n,seq_len,nb_feat], y [n,1]
    """
    # https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816
    # read all vo2, generate xtrain,ytrain,xtest,ytest
    # note: don't use overlapping participants
    # vo2_data = df['vo2'].values

    # (x,y) is (N,L,F) and (N,1), N=#sequences, L=sequence length, F=#feature channels

    # https://www.tensorflow.org/guide/data
    # https://www.tensorflow.org/tutorials/structured_data/time_series
    # https://www.tensorflow.org/tutorials/load_data/csv
    # If all of your input data fits in memory, the simplest way to create a Dataset from them is to convert them to tf.Tensor objects and use Dataset.from_tensor_slices().

    xtrain, xtrain_static, ytrain, train_descr = \
        _get_x_y_helper(seq_len, feature_list, seq_step=seq_step_train, data_type='train', vo2_type=vo2_type)
    # Get every sequence in the data for testing
    xval, xval_static, yval, val_descr = \
        _get_x_y_helper(seq_len, feature_list, seq_step=1, data_type='validation', vo2_type=vo2_type)
    xtest, xtest_static, ytest, test_descr = \
        _get_x_y_helper(seq_len, feature_list, seq_step=1, data_type='test', vo2_type=vo2_type)

    # TODO: change to estimators? https://towardsdatascience.com/how-to-normalize-features-in-tensorflow-5b7b0e3a4177
    mu = np.mean(xtrain[:, 0, :], axis=0)
    sigma = np.std(xtrain[:, 0, :], axis=0)
    min_ = np.min(xtrain[:, 0, :], axis=0)
    max_ = np.max(xtrain[:, 0, :], axis=0)
    if norm_type == 'norm':
        numer = min_
        denom = max_ - min_
    elif norm_type == 'stand':
        numer = mu
        denom = sigma
    else:
        numer = mu
        denom = sigma
        if 'WR' in feature_list:
            idx = feature_list.index('WR')
            numer[idx] = min_[idx]
            denom[idx] = max_[idx] - min_[idx]
    xtrain = (xtrain - numer) / denom
    xval = (xval - numer) / denom
    xtest = (xtest - numer) / denom
    # xtrain = (xtrain-min_)/(max_-min_)
    # xval = (xval-min_)/(max_-min_)
    # xtest = (xtest-min_)/(max_-min_)

    train = {'x': xtrain, 'static': xtrain_static, 'y': ytrain, 'descr': train_descr}
    val = {'x': xval, 'static': xval_static, 'y': yval, 'descr': val_descr}
    test = {'x': xtest, 'static': xtest_static, 'y': ytest, 'descr': test_descr}

    return train, val, test


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