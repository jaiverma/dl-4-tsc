import tensorflow as tf
import keras
import sys

model = keras.models.load_model(sys.argv[1])
print(model.output.op.name)

saver = tf.train.Saver()
saver.save(keras.backend.get_session(), '{}/keras_model.ckpt'.format(sys.argv[2]))

'''
python freeze_graph.py --input_meta_graph=/local/mnt/workspace/playground/makers/cnn/2d/keras_model/keras_model.ckpt.meta --input_checkpoint=/local/mnt/workspace/playground/makers/cnn/2d/keras_model/keras_model.ckpt --output_graph=/local/mnt/workspace/playground/makers/cnn/2d/frozen_graph.pb --output_node_names="dense_1/Sigmoid" --input_binary=true
'''
