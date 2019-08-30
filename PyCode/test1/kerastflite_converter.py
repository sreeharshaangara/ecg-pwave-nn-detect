import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file(".\current_run.h5")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
