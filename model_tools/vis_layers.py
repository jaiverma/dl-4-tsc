import sys
import keras
import numpy as np

def read_dataset(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

model = keras.models.load_model(sys.argv[1]) # best_model.hdf5

def get_layer_out(layer_name=None):
    if layer_name is None:
        outputs = [layer.output for layer in model.layers][1:]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    if len(outputs) == 0:
        raise RuntimeError('no layer with name: {}'.format(layer_name))
    return keras.backend.function(
        [model.input, keras.backend.learning_phase()],
        outputs
    )

x_val, y_val = read_dataset(sys.argv[2]) # pong_TEST
# x_val = x_val[1] # run for just 1 entry

if sys.argv[3] == '1d':
    x_val = x_val.reshape(x_val.shape + (1,))

    # flatten_output = np.array(get_layer_out('flatten_1')([x_val, 1])).astype(np.float32)
    # flatten_output.tofile('flatten.raw')
    reshape_output = np.array(get_layer_out('reshape_1')([x_val, 1])).astype(np.float32)
    reshape_output.tofile('reshape.raw')
    # layer_output = get_layer_out()([x_val, 1])
    # conv1d_1_output = np.array(get_layer_out('conv1d_1')([x_val, 1])).astype(np.float32)
    # conv1d_2_output = np.array(get_layer_out('conv1d_2')([x_val, 1])).astype(np.float32)
    # conv1d_1_entry_0_output = conv1d_1_output[0][0]
    # conv1d_2_entry_0_output = conv1d_2_output[0][0]
    # conv1d_1_entry_0_output.tofile('conv1d_1_output.raw')
    # conv1d_2_entry_0_output.tofile('conv1d_2_output.raw')

elif sys.argv[3] == '2d':
    x_val = x_val.reshape(x_val.shape + (1, 1))

    # layer_output = get_layer_out()([x_val, 1])
    conv2d_1_output = np.array(get_layer_out('conv2d_1')([x_val, 1])).astype(np.float32)
    conv2d_2_output = np.array(get_layer_out('conv2d_2')([x_val, 1])).astype(np.float32)
    conv2d_1_entry_0_output = conv2d_1_output[0][0]
    conv2d_2_entry_0_output = conv2d_2_output[0][0]
    conv2d_1_entry_0_output.tofile('conv2d_1_output.raw')
    conv2d_2_entry_0_output.tofile('conv2d_2_output.raw')

'''
Model with 1d-convolution

>>> model.summary()
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 50, 1)             0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 50, 6)             48
_________________________________________________________________
average_pooling1d_1 (Average (None, 16, 6)             0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 16, 12)            516
_________________________________________________________________
average_pooling1d_2 (Average (None, 5, 12)             0
_________________________________________________________________
flatten_1 (Flatten)          (None, 60)                0
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 122
=================================================================
Total params: 686
Trainable params: 686
Non-trainable params: 0
_________________________________________________________________
'''

'''
Model with 2d-convoltion

Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 50, 1, 1)          0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 50, 1, 6)          48
_________________________________________________________________
average_pooling2d_1 (Average (None, 16, 1, 6)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 1, 12)         516
_________________________________________________________________
average_pooling2d_2 (Average (None, 5, 1, 12)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 60)                0
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 122
=================================================================
Total params: 686
Trainable params: 686
Non-trainable params: 0
_________________________________________________________________
'''
