#!/usr/bin/env python3
print('testing TensorFlow...')
import tensorflow as tf

print('TensorFlow version: ' + str(tf.__version__))
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

#
# TensorFlow v1.x tests
#
if str(tf.__version__).find("1.") == 0:
    import urllib
    import tarfile
    import time
    import os
    
    # create a test session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # https://github.com/tensorflow/models/blob/r1.13.0/research/object_detection/object_detection_tutorial.ipynb
    # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
    MODEL_DIR = '/tmp'
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
    
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_DIR, MODEL_NAME, 'frozen_inference_graph.pb')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    
    # Download and extract the model
    print(f'downloading {DOWNLOAD_BASE + MODEL_FILE}')
    
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_PATH)
    tar_file = tarfile.open(MODEL_PATH)
    
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, MODEL_DIR)  # os.getcwd()
   
    print(f'loading {PATH_TO_FROZEN_GRAPH}')
    
    # Load a (frozen) Tensorflow model into memory
    begin_time = time.perf_counter()
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')   
    
    end_time = time.perf_counter()
    print(f'loaded {PATH_TO_FROZEN_GRAPH} in {end_time - begin_time} seconds')
    
print('TensorFlow OK\n')
