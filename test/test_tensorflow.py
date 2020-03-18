
print('testing TensorFlow...')
import tensorflow as tf

print('TensorFlow version: ' + str(tf.__version__))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print('TensorFlow OK\n')
