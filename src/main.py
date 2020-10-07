import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input, Embedding
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from tensorflow.keras import Model, Input, optimizers
import numpy as np
from datetime import datetime
import util
import pandas as pd
import argparse

from tcn import TCN

# # Disables GPU
# tf.config.set_visible_devices([], 'GPU')
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
parser.add_argument('--epochs', default=30, type=int, help='training epochs')
parser.add_argument('--note', default='p08 test', type=str, help='note to log with model')
parser.add_argument('--log_dir', default='logs/', type=str, help='tensorboard log dir')
parser.add_argument('--chkpt_dir', default='chkpts/', type=str, help='tensorflow checkpoint dir')
args = parser.parse_args()

# TCN parameters
use_demographics = args.use_demographics
nb_filters = args.nb_filters
kernel_size = args.kernel_size
dilations = [2 ** i for i in range(args.max_dilation_pow+1)]
dropout_rate = args.dropout_rate
batch_size = args.batch_size
epochs = args.epochs
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


class PrintSomeValues(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print('y_true, y_pred')
        if use_demographics:
            pred = self.model.predict([x_val[:5], x_val_static[:5]])
        else:
            pred = self.model.predict(x_val[:5])
        print(np.hstack([y_test[:5], pred]))


def build_model():
    # "The most important factor for picking parameters is to make sure that
    # the TCN has a sufficiently large receptive field by choosing k and d
    # that can cover the amount of context needed for the task." (Bai 2018)
    # Receptive field = nb_stacks * kernel_size * last_dilation
    num_feat = x_train.shape[2]
    max_len = x_train.shape[1]
    lr = 0.002
    input_layer = Input(shape=(max_len, num_feat))
    x = TCN(nb_filters, kernel_size, 1, dilations, 'causal',
            False, dropout_rate, False,
            'relu', 'he_normal', False, True,
            name='tcn')(input_layer)

    input_layers = [input_layer]
    if use_demographics:
        nb_static = x_train_static.shape[1]
        # TODO: use Embedding layer with one-hot indexing (see Esteban 2015/2016), or use directly.
        input_layer_static = Input(shape=(nb_static,))
        # y = Embedding(nb_static, nb_static)(input_layer_static)
        x = tf.keras.layers.concatenate([x, input_layer_static])
        input_layers.append(input_layer_static)

    z = Dense(1)(x)
    z = Activation('linear')(z)
    output_layer = z
    model = Model(input_layers, output_layer)
    opt = optimizers.Adam(lr=lr, clipnorm=1.)
    model.compile(opt, loss='mean_squared_error')
    print('model.x = {}'.format([l.shape for l in input_layers]))
    print('model.y = {}'.format(output_layer.shape))
    # model = compiled_tcn(return_sequences=False,
    #                      num_feat=x_train.shape[2],
    #                      num_classes=0,
    #                      nb_filters=nb_filters,
    #                      kernel_size=kernel_size,
    #                      dilations=dilations,
    #                      nb_stacks=1,
    #                      max_len=x_train.shape[1],
    #                      use_skip_connections=False,
    #                      regression=True,
    #                      dropout_rate=dropout_rate,
    #                      use_layer_norm=True)
    return model


def run_task():
    model = build_model()

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
    # mdl_dir = 'models/mdl' + timestamp
    # print('Saving model {}...'.format(mdl_dir))
    # model.save(mdl_dir)
    # print('Done')

    # Save data and network options for repeatability/understandability.
    opts = vars(args)
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
        fig.show()
        fig.write_html(chkpt_dir + '/plotly_{}.html'.format(descriptor))

    # TODO: load best model weights before predicting
    yhat_val = model.predict(x_val_all)
    yhat_test = model.predict(x_test_all)
    y_yhat_save_csv_plotly(y_val, yhat_val, 'val')
    y_yhat_save_csv_plotly(y_test, yhat_test, 'test')


if __name__ == '__main__':
    run_task()
