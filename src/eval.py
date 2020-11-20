import pandas as pd
import numpy as np
import util
import re
import os
import sys
import tensorflow as tf
import timeit

# LSTM fails without this
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_test_data(chkpt_dir, norm_type='mixed'):
    csvfile = chkpt_dir + '/network_params.csv'
    params = pd.read_csv(csvfile, header=None, index_col=0, squeeze=True).to_dict()
    seq_len = int(params['seq_len'])
    feature_list = params['feature_list'].split(',')
    seq_step_train = int(params['seq_step_train'])
    vo2_type = params['vo2_type']
    _, val, test = util.get_x_y(seq_len, feature_list, seq_step_train, vo2_type, norm_type)
    return val, test


def extract_loss_mets_to_csv_zignoli(chkpt_dir):
    val, test = get_test_data(chkpt_dir, 'stand')
    x_test = test['x']
    y_test = test['y']
    x_static_test = test['static']
    descr_test = test['descr']
    x_val = val['x']
    y_val = val['y']
    test_kg = x_static_test[:, 1]
    opts = {'max_len': x_test.shape[1],
            'nb_feat': x_test.shape[2],
            'lr': 0.0005}  # should read from network_params.csv, but I don't think this matters for eval
    model = util.build_zignoliLSTM(opts)
    model.load_weights(chkpt_dir + '/')
    yhat_val = model.predict(x_val)
    mae = np.mean(np.abs(yhat_val - y_val))
    mse = np.mean(np.square(yhat_val - y_val))  # loss

    t0 = timeit.default_timer()
    yhat_test = model.predict(x_test)
    dt = timeit.default_timer() - t0
    print(f'Time to predict x_test with shape {x_test.shape}: {dt}')

    vo2 = np.squeeze(y_test)
    vo2_hat = np.squeeze(yhat_test)
    mets = vo2 / test_kg * 1000 / 3.5
    mets_hat = vo2_hat / test_kg * 1000 / 3.5
    protocol = descr_test[:, 0]
    pid = descr_test[:, 1]
    best_results = np.column_stack((protocol, pid, vo2, vo2_hat, mets, mets_hat))

    parts = path.split(os.sep)
    chkpt = parts[-2]
    df = pd.DataFrame([(mse, mae)], columns=['mse_val', 'mae_val'])
    df.to_csv(f'../eval/{chkpt}_val_zignoli.csv', index=False)

    df = pd.DataFrame(best_results, columns=['protocol', 'pid', 'vo2', 'vo2_hat', 'mets', 'mets_hat'])
    df.to_csv(f'../eval/{chkpt}_test_zignoli.csv', index=False)


def extract_loss_mets_to_csv(chkpt_parent):
    # chkpt_parent has children folder structure fx_kx_dx_drx_lrx, each of which has log<date>
    metrics = []
    best_loss = 999
    nb_models = len(os.listdir(chkpt_parent))
    trials_skipped = []
    trial_mse = []
    for i, fx_kx_dx_drx_lrx in enumerate(os.scandir(chkpt_parent)):
        sys.stdout.write(f'\r{i}/{nb_models} {fx_kx_dx_drx_lrx.name}')
        sys.stdout.flush()
        m = re.search('f(\d+)_k(\d)_d(\d)_dr(0.\d)_lr(0.\d+)', fx_kx_dx_drx_lrx.name)
        nb_filters = int(m.group(1))
        kernel_size = int(m.group(2))
        max_dilation_pow = int(m.group(3))
        dropout_rate = float(m.group(4))
        lr = float(m.group(5))
        receptive_field = kernel_size * 2**max_dilation_pow if kernel_size > 1 else 1

        # Should be only 1 subdirectory (chkptyyyymmdd-hhmmss)
        chkpt_dir = next(os.scandir(fx_kx_dx_drx_lrx.path)).path
        try:
            val, test = get_test_data(chkpt_dir)
        except FileNotFoundError:
            print(f'{fx_kx_dx_drx_lrx.name} network_params.csv not found. Skipping.')
            trials_skipped.append(fx_kx_dx_drx_lrx.name)
            continue
        x_test = test['x']
        y_test = test['y']
        x_static_test = test['static']
        descr_test = test['descr']
        x_val = val['x']
        y_val = val['y']
        test_kg = x_static_test[:, 1]
        opts = {'max_len': x_test.shape[1],
                'nb_feat': x_test.shape[2],
                'nb_filters': nb_filters,
                'kernel_size': kernel_size,
                'dilations': [2 ** i for i in range(max_dilation_pow+1)],
                'dropout_rate': dropout_rate,
                'lr': lr}
        # TODO: model doesn't support demographics like this
        model = util.build_model(opts)
        model.load_weights(chkpt_dir + '/')
        model.summary()
        yhat_val = model.predict(x_val)
        mae = np.mean(np.abs(yhat_val - y_val))
        mse = np.mean(np.square(yhat_val - y_val))  # loss
        metrics.append((nb_filters, kernel_size, 2**max_dilation_pow, receptive_field, mse, mae))
        trial_mse.append((fx_kx_dx_drx_lrx.name, mse))

        if mse < best_loss:
            best_model_name = fx_kx_dx_drx_lrx.name
            best_loss = mse
            t0 = timeit.default_timer()
            yhat_test = model.predict(x_test)
            dt = timeit.default_timer() - t0
            print(f'Time to predict x_test with shape {x_test.shape}: {dt}')

            vo2 = np.squeeze(y_test)
            vo2_hat = np.squeeze(yhat_test)
            mets = vo2 / test_kg * 1000 / 3.5
            mets_hat = vo2_hat / test_kg * 1000 / 3.5
            protocol = descr_test[:, 0]
            pid = descr_test[:, 1]
            best_results = np.column_stack((protocol, pid, vo2, vo2_hat, mets, mets_hat))

    for trial in trials_skipped:
        print(f'Skipped {trial}')
    trial_mse.sort(key=lambda x: x[1])
    for x in trial_mse:
        print(f'{x[0]},{x[1]}')

    head, tail = chkpt_parent, ''
    while re.search('(\d{8})', tail) is None:
        head, tail = os.path.split(head)
    jobid = tail
    df = pd.DataFrame(metrics, columns=['nb_filters', 'kernel_size', 'max_dilation', 'receptive_field', 'mse_val', 'mae_val'])
    df.to_csv(f'../eval/{jobid}_val.csv', index=False)

    df = pd.DataFrame(best_results, columns=['protocol', 'pid', 'vo2', 'vo2_hat', 'mets', 'mets_hat'])
    df.to_csv(f'../eval/{jobid}_test_{best_model_name}.csv', index=False)


if __name__ == '__main__':
    zignoli_trials = ['39735459', '39735466', '39752794']
    print('Input checkpoint parent directory: ')
    chkpt_parent = input()

    path = os.path.normpath(chkpt_parent)
    parts = path.split(os.sep)
    if any(trial in parts for trial in zignoli_trials):
        extract_loss_mets_to_csv_zignoli(path)
    else:
        extract_loss_mets_to_csv(path)
