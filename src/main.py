import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input, Embedding
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras import Model, Input, optimizers
import numpy as np
from datetime import datetime
import util
import pandas as pd

from tcn import TCN

# # TODO: disabling GPU
# tf.config.set_visible_devices([], 'GPU')

seq_len = 20
feature_list = ['WR', 'HR', 'VE', 'BF', 'HRR']
seq_step_train = 5
vo2_type = 'VO2'
x_train, x_train_static, y_train, x_val, x_val_static, y_val, x_test, x_test_static, y_test = \
    util.get_x_y(seq_len, feature_list, seq_step_train, vo2_type)
print('x_train: {}\nx_val: {}\nx_test: {}'.format(x_train.shape, x_val.shape, x_test.shape))
print('seq_len: {}\nfeature_list: {}'.format(seq_len, feature_list))

# TCN parameters
nb_filters = 24
kernel_size = 8
dilations = [2 ** i for i in range(4)]
dropout_rate = 0.2
batch_size = 32
epochs = 25
note = 'with demographics (first time)'  # note describing the setup


class PrintSomeValues(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print('y_true, y_pred')
        print(np.hstack([y_test[:5], self.model.predict([x_test[:5], x_test_static[:5]])]))


def run_task():
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

    # Add static demographic info
    nb_static = x_train_static.shape[1]
    # TODO: use Embedding layer with one-hot indexing, or use directly.
    input_layer_static = Input(shape=(nb_static,))
    #y = Embedding(nb_static, nb_static)(input_layer_static)

    xy = tf.keras.layers.concatenate([x, input_layer_static])

    z = Dense(1)(xy)
    z = Activation('linear')(z)
    output_layer = z
    model = Model([input_layer, input_layer_static], output_layer)
    opt = optimizers.Adam(lr=lr, clipnorm=1.)
    model.compile(opt, loss='mean_squared_error')
    print('model.x = {}, {}'.format(input_layer.shape, input_layer_static.shape))
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

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')

    psv = PrintSomeValues()

    # Using sparse softmax.
    # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
    model.summary()

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir = 'logs/log' + timestamp
    tensorboard = TensorBoard(log_dir=logdir, update_freq='epoch', profile_batch=0)

    history = model.fit([x_train, x_train_static], y_train, validation_data=([x_val, x_val_static], y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[psv, tensorboard])
    mdl_dir = 'models/mdl' + timestamp
    model.save(mdl_dir)

    # Save network options for repeatability/understandability.
    opts = {'seq_len': seq_len, 'feature_list': feature_list, 'seq_step_train': seq_step_train, 'vo2_type': vo2_type,
            'nb_filters': nb_filters, 'kernel_size': kernel_size, 'dilations': dilations, 'dropout_rate': dropout_rate,
            'batch_size': batch_size, 'epochs': epochs, 'note': note}
    csv_file = mdl_dir + '/network_params.csv'
    pd.DataFrame.from_dict(data=opts, orient='index').to_csv(csv_file, header=False)

    #ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
    #manager = tf.train.CheckpointManager(ckpt, './ckpts', max_to_keep=3)

    #model.evaluate(x_test, y_test, verbose=2)

    pred = model.predict([x_val, x_val_static])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    #fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_val.flatten(), line_color='rgb(0.2,0.2,0.2)', name='y'))
    fig.add_trace(go.Scatter(y=pred.flatten(), line_color='rgba(255,0,0,0.8)', name='y_hat'))
    fig.update_layout(title=mdl_dir)
    fig.show()
    fig.write_html(mdl_dir + '/plotly.html')


if __name__ == '__main__':
    run_task()
