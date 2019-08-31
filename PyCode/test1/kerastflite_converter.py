import tensorflow as tf


model = tf.keras.models.load_model(".\current_run.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

