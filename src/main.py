# TODO: install GPU requirements for TF 2.3.0

from tensorflow.keras.callbacks import Callback
import numpy as np
import util

from tcn import compiled_tcn


def data_generator(n, seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        n: # of data in the set
    """
    x_num = np.random.uniform(0, 1, (n, 1, seq_length))
    x_mask = np.zeros([n, 1, seq_length])
    y = np.zeros([n, 1])
    for i in range(n):
        positions = np.random.choice(seq_length, size=2, replace=False)
        x_mask[i, 0, positions[0]] = 1
        x_mask[i, 0, positions[1]] = 1
        y[i, 0] = x_num[i, 0, positions[0]] + x_num[i, 0, positions[1]]
    x = np.concatenate((x_num, x_mask), axis=1)
    x = np.transpose(x, (0, 2, 1))
    return x, y


def get_x_y(size=1000):
    x_train, y_train = data_generator(n=200000, seq_length=600)
    x_test, y_test = data_generator(n=40000, seq_length=600)
    return x_train, y_train, x_test, y_test


# x_train, y_train, x_test, y_test = get_x_y()
x_train, y_train, x_test, y_test = util.get_x_y()


class PrintSomeValues(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print('y_true, y_pred')
        print(np.hstack([y_test[:5], self.model.predict(x_test[:5])]))


def run_task():
    model = compiled_tcn(return_sequences=False,
                         num_feat=x_train.shape[2],
                         num_classes=0,
                         nb_filters=24,
                         kernel_size=8,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         max_len=x_train.shape[1],
                         use_skip_connections=False,
                         regression=True,
                         dropout_rate=0.2,
                         use_layer_norm=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')

    psv = PrintSomeValues()

    # Using sparse softmax.
    # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
    model.summary()

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15,
              batch_size=256, callbacks=[psv])


if __name__ == '__main__':
    run_task()
