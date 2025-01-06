from Detector import *

modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
# modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz'
# modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz'
classFile = 'coco.names'
imagePath = 'image/test.jpeg'

threshold = 0.5

detection = Detector()
detection.readClasses(classFile)
detection.downloadModel(modelURL)
detection.loadModel()
detection.predictImage(imagePath, threshold)