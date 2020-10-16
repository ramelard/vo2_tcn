import pandas as pd
import numpy as np
import util
import re
import os


def get_test_data(chkpt_dir):
    csvfile = chkpt_dir + '/network_params.csv'
    params = pd.read_csv(csvfile, header=None, index_col=0, squeeze=True).to_dict()
    seq_len = int(params['seq_len'])
    feature_list = params['feature_list'].split(',')
    seq_step_train = int(params['seq_step_train'])
    vo2_type = params['vo2_type']
    _, _, _, x_val, x_val_static, y_val, x_test, x_test_static, y_test = \
        util.get_x_y(seq_len, feature_list, seq_step_train, vo2_type)
    # TODO: NEED TO NORMALIZE ACCORDING TO TRAINING SET. MAYBE JUST SAVE DATA.
    return x_val, y_val, x_test, y_test


def hyperparam_loss_to_csv(chkpt_parent):
    # chkpt_parent has children folder structure fx_kx_dx_drx_lrx, each of which has log<date>
    metrics = []

    for fx_kx_dx_drx_lrx in os.scandir(chkpt_parent):
        m = re.search('f(\d+)_k(\d)_d(\d)_dr(0.\d)_lr0.0005', fx_kx_dx_drx_lrx.name)
        nb_filters = int(m.group(1))
        kernel_size = int(m.group(2))
        max_dilation_pow = int(m.group(3))
        dropout_rate = float(m.group(4))
        receptive_field = kernel_size * max_dilation_pow

        # Should be only 1 subdirectory (chkptyyyymmdd-hhmmss)
        chkpt_dir = next(os.scandir(fx_kx_dx_drx_lrx.path)).path
        x_val, y_val, x_test, y_test = get_test_data(chkpt_dir)
        opts = {'max_len': x_test.shape[1],
                'num_feat': x_test.shape[2],
                'nb_filters': nb_filters,
                'kernel_size': kernel_size,
                'dilations': [2 ** i for i in range(max_dilation_pow+1)],
                'dropout_rate': dropout_rate}
        # TODO: model doesn't support demographics like this
        model = util.build_model(opts)
        model.load_weights(chkpt_dir + '/')
        yhat_test = model.predict(x_test)
        mae = np.mean(np.abs(yhat_test - y_test))
        mse = np.mean(np.square(yhat_test - y_test))  # aka loss
        metrics.append((receptive_field, mse, mae))

    df = pd.DataFrame(metrics, columns=['receptive_field', 'mse', 'mae'])
    df.to_csv(f'../eval/{chkpt_parent}.csv', index=False)


if __name__ == '__main__':
    print('Input checkpoint parent directory: ')
    chkpt_parent = input()
    hyperparam_loss_to_csv(chkpt_parent)