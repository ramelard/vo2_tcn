import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Lambda, concatenate



class TemporalBlock(tf.keras.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = Conv1D(filters=n_outputs,
                            kernel_size=kernel_size,
                            padding='causal',
                            dilation_rate=dilation_rate,
                            strides=strides,
                            activation=tf.nn.relu,
                            name='conv1')
        self.conv2 = Conv1D(filters=n_outputs,
                            kernel_size=kernel_size,
                            padding='causal',
                            dilation_rate=dilation_rate,
                            strides=strides,
                            activation=tf.nn.relu,
                            name='conv2')
        #self.conv1 = CausalConv1D(
        #    n_outputs, kernel_size, strides=strides,
        #    dilation_rate=dilation_rate, activation=tf.nn.relu,
        #    name="conv1")
        # self.conv2 = CausalConv1D(
        #     n_outputs, kernel_size, strides=strides,
        #     dilation_rate=dilation_rate, activation=tf.nn.relu,
        #     name="conv2")
        self.down_sample = None

    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        if input_shape[channel_dim] != self.n_outputs:
            # self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1,
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)

    def call(self, inputs, training=True):
        #x = self.conv1(inputs)
        #x = tf.contrib.layers.layer_norm(x)
        x = tfa.layers.WeightNormalization(self.conv1(inputs))
        x = self.dropout1(x, training=training)
        x = tfa.layers.WeightNormalization(self.conv2(x))
        #x = self.conv2(x)
        #x = tf.contrib.layers.layer_norm(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)
