import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input, Embedding
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from tensorflow.keras import Model, Input, optimizers
import numpy as np
import pandas as pd
from simpleflock import SimpleFlock  # TODO: LINUX ONLY
from datetime import datetime
import util
import argparse
import os
import warnings

from tcn import TCN

# # Disables GPU
# tf.config.set_visible_devices([], 'GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#tf.debugging.set_log_device_placement(True)

parser = argparse.ArgumentParser(description='VO2 TCN Prediction')
parser.add_argument('--use_demographics', default=False, action='store_true', help='use static demographic info in training')
parser.add_argument('--seq_len', type=int, help='sequence length')
parser.add_argument('--feature_list', default='WR,HR,VE,BF,HRR', type=str, help='comma separated list of features from {WR, HR, VE, BF, HRR}')
parser.add_argument('--seq_step_train', default=3, type=int, help='steps in between training sequences')
parser.add_argument('--vo2_type', default='VO2', type=str, help='VO2 or VO2rel')
parser.add_argument('--nb_filters', default=24, type=int, help='number of conv1d filters')
parser.add_argument('--kernel_size', default=4, type=int, help='TCN kernel size')
parser.add_argument('--max_dilation_pow', default=5, type=int, help='maximum dilation power specified as x where (2^x)')
parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout rate')
parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
parser.add_argument('--epochs', default=50, type=int, help='training epochs')
parser.add_argument('--note', default='', type=str, help='note to log with model')
parser.add_argument('--log_dir', default='logs/', type=str, help='tensorboard log dir')
parser.add_argument('--chkpt_dir', default='chkpts/', type=str, help='tensorflow checkpoint dir')
parser.add_argument('--gpu', default=0, type=int, help='gpu device to use')
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
args = parser.parse_args()

#tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
#if tf.config.list_physical_devices('GPU'):
#    distributed_strategy = tf.distribute.MirroredStrategy()
#else:
#    distributed_strategy = tf.distribute.get_strategy()
distributed_strategy = tf.distribute.get_strategy()
# TODO: only run this on graham
nb_gpus = len(gpus)
gpu_idx = 0
# TODO: suboptimal way to do things. better to release gpu_idx as it's finished.
if nb_gpus > 1:
    local_scratch = os.getenv('SLURM_TMPDIR')
    with SimpleFlock(local_scratch + '/.gpulock', timeout=10):
        with open(local_scratch + '/gpuidx', 'r+') as f:
            i = f.read()
            gpu_idx = int(i) if i else 0
            f.seek(0)
            if gpu_idx > 9:  # TODO
                warnings.warn('WARNING: double digit GPUs not supported properly')
            f.write(str((gpu_idx+1) % nb_gpus))
print(f'Using /GPU:{gpu_idx}')
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue 152
#os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

# TCN parameters
use_demographics = args.use_demographics
nb_filters = args.nb_filters
kernel_size = args.kernel_size
dilations = [2 ** i for i in range(args.max_dilation_pow+1)]
dropout_rate = args.dropout_rate
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
note = args.note  # note describing the setup

# Dataset parameters
if args.seq_len is None:
    args.seq_len = kernel_size * dilations[-1]
seq_len = args.seq_len
feature_list = [s for s in args.feature_list.split(',')]
seq_step_train = args.seq_step_train
vo2_type = args.vo2_type
x_train, x_train_static, y_train, x_val, x_val_static, y_val, x_test, x_test_static, y_test = \
    util.get_x_y(seq_len, feature_list, seq_step_train, vo2_type)
print('x_train: {}\nx_val: {}\nx_test: {}'.format(x_train.shape, x_val.shape, x_test.shape))
print('seq_len: {}\nfeature_list: {}'.format(seq_len, feature_list))

if use_demographics:
    x_train_all = [x_train, x_train_static]
    x_val_all = [x_val, x_val_static]
    x_test_all = [x_test, x_test_static]
else:
    x_train_all = x_train
    x_val_all = x_val
    x_test_all = x_test

# Print initial GPU state
os.system('nvidia-smi')


class PrintSomeValues(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print('y_true, y_pred')
        if use_demographics:
            pred = self.model.predict([x_val[:5], x_val_static[:5]])
        else:
            pred = self.model.predict(x_val[:5])
        print(np.hstack([y_test[:5], pred]))
        os.system('nvidia-smi')


def run_task():
    # model = build_model()
    opts = vars(args)
    opts['dilations'] = dilations
    model = util.build_model(opts, use_demographics=use_demographics, nb_static=x_train_static.shape[1])

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')

    # Using sparse softmax.
    # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
    model.summary()

    # Set up callbacks
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = args.log_dir + '/log' + timestamp
    tensorboard = TensorBoard(log_dir=log_dir, update_freq='epoch', profile_batch=0)
    chkpt_dir = args.chkpt_dir + '/chkpt{}/'.format(timestamp)
    chkpt = ModelCheckpoint(filepath=chkpt_dir, # + 'epoch{epoch:02d}',
                            save_best_only=True,
                            save_weights_only=True,  # False->model.save
                            verbose=1)
    psv = PrintSomeValues()

    # To load weights:
    # model = build_model
    # model.load_weights(chkpt_dir + 'epoch05')

    # Train!
    history = model.fit(x_train_all, y_train, validation_data=(x_val_all, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[psv, tensorboard, chkpt])
    print('Model fit complete. Saving variables.')
    # mdl_dir = 'models/mdl' + timestamp
    # print('Saving model {}...'.format(mdl_dir))
    # model.save(mdl_dir)
    # print('Done')

    # Save data and network options for repeatability/understandability.
    csv_file = chkpt_dir + '/network_params.csv'
    pd.DataFrame.from_dict(data=opts, orient='index').to_csv(csv_file, header=False)

    def y_yhat_save_csv_plotly(y, yhat, descriptor):
        y_yhat = np.concatenate((y, yhat), axis=1)
        df = pd.DataFrame(y_yhat)
        csv_filename = 'y_yhat_{}.csv'.format(descriptor)
        df.to_csv(chkpt_dir + '/' + csv_filename, header=['y', 'yhat'], index=False)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        #fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y.flatten(), line_color='rgb(0.2,0.2,0.2)', name='y'))
        fig.add_trace(go.Scatter(y=yhat.flatten(), line_color='rgba(255,0,0,0.8)', name='y_hat'))
        fig.update_layout(title=chkpt_dir + '({})'.format(descriptor))
        # fig.show()
        fig.write_html(chkpt_dir + '/plotly_{}.html'.format(descriptor))

    # Load best model weights before predicting
    print('Generating prediction results')
    model.load_weights(chkpt_dir + '/')
    yhat_val = model.predict(x_val_all)
    yhat_test = model.predict(x_test_all)
    y_yhat_save_csv_plotly(y_val, yhat_val, 'val')
    y_yhat_save_csv_plotly(y_test, yhat_test, 'test')


if __name__ == '__main__':
    with tf.device(f'GPU:{gpu_idx}'):
        run_task()
