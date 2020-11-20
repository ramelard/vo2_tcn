import os
import re
import pandas as pd
import numpy as np
import plotly.express as px
from tcn import tcn_full_summary
import util


# Read in an existing f_k_d_dr_lr folder checkpoint, load its weights into a TCN.
# Change the input variables by time index one by one to confirm that only changes within our receptive field is
# changing the estimation.

chkpt_parent = r'C:\Users\ramelard\Downloads\temp2'
fx_kx_dx_drx_lrx = next(os.scandir(chkpt_parent))
chkpt_dir = next(os.scandir(fx_kx_dx_drx_lrx.path)).path

m = re.search('f(\d+)_k(\d)_d(\d)_dr(0.\d)_lr(0.\d+)', fx_kx_dx_drx_lrx.name)
nb_filters = int(m.group(1))
kernel_size = int(m.group(2))
max_dilation_pow = int(m.group(3))
dropout_rate = float(m.group(4))
lr = float(m.group(5))

def get_test_data(chkpt_dir, norm_type='mixed'):
    csvfile = chkpt_dir + '/network_params.csv'
    params = pd.read_csv(csvfile, header=None, index_col=0, squeeze=True).to_dict()
    seq_len = int(params['seq_len'])
    feature_list = params['feature_list'].split(',')
    seq_step_train = int(params['seq_step_train'])
    vo2_type = params['vo2_type']
    _, val, test = util.get_x_y(seq_len, feature_list, seq_step_train, vo2_type, norm_type)
    return val, test

val, test = get_test_data(chkpt_dir)
x_test = test['x']
y_test = test['y']

opts = {'max_len': x_test.shape[1],
        'nb_feat': x_test.shape[2],
        'nb_filters': nb_filters,
        'kernel_size': kernel_size,
        'dilations': [2 ** i for i in range(max_dilation_pow + 1)],
        'dropout_rate': dropout_rate,
        'lr': lr}

model = util.build_model(opts)
model.load_weights(chkpt_dir + '/')
tcn_full_summary(model)

# tf.debugging.experimental.enable_dump_debug_info(
#    "../logs/dilations/",
#    tensor_debug_mode="FULL_HEALTH",
#    circular_buffer_size=-1)

for rb in model.layers[1].residual_blocks:
    print(f'{rb.name}: ', end="")
    [l.dilation_rate for l in rb.layers if 'conv1D' in l.name]

d = np.expand_dims(x_test[0], axis=0)
# d = np.concatenate((np.zeros((1,24,5)), d), axis=1)
a = model.predict(d)
vals = []
T = d.shape[1]
for idx in range(T - 1, -1, -1):
    # for idx in range(-2, -100, -1):
    dcopy = np.copy(d)
    dcopy[0, idx, :] = 10  # 1-dcopy[0,idx,:]
    aprime = model.predict(dcopy)
    vals.append(float(aprime[0]))
    # print(f'{i}: ({a}, {aprime})')
    print(f'\r{idx}/{T}', end="")

# print(vals)

fig = px.line(vals)
fig.show()
