
print('testing TensorFlow...')
import tensorflow as tf

print('TensorFlow version: ' + str(tf.__version__))
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

if str(tf.__version__).find("1.") == 0:
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print('TensorFlow OK\n')
