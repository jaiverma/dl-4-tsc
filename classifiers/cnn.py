# Time-CNN
import keras
import numpy as np
import time
import tensorflow as tf

from utils.utils import save_logs


class Classifier_CNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        if (verbose == True):
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = keras.layers.Input((50,1,1))

        if input_shape[0] < 60: # for italypowerondemand dataset
            padding = 'same'

        conv1 = keras.layers.Conv2D(filters=6,kernel_size=(7,1),padding=padding,activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling2D(pool_size=(3,1))(conv1)

        conv2 = keras.layers.Conv2D(filters=12,kernel_size=(7,1),padding=padding,activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling2D(pool_size=(3,1))(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [model_checkpoint]

        return model

    # def build_model(self, input_shape, nb_classes):
    #     padding = 'same'
    #     input_layer = tf.placeholder(tf.float32, shape=(50, 1))
    #     conv1 = tf.layers.conv1d(input_layer, filters=6, kernel_size=7, padding=padding, activation='sigmoid')
    #     conv1 = tf.layers.average_pooling1d(conv1, pool_size=3)

    #     conv2 = tf.layers.conv1d(conv1, filters=12, kernel_size=7, padding=padding, activation='sigmoid')
    #     conv2 = tf.layers.average_pooling1d(conv2, pool_size=3)

    #     flatten_layer = tf.layers.flatten(conv2)

    #     output_layer = tf.layers.dense(flatten_layer, units=nb_classes, activation='sigmoid')

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        mini_batch_size = 16
        nb_epochs = 2000

        start_time = time.time()

        x_train = x_train.reshape(290, 50, 1, 1)
        x_val = x_val.reshape(82, 50, 1, 1)

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration,lr=False)

        keras.backend.clear_session()


