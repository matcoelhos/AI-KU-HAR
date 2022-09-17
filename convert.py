import tensorflow as tf

model_list = ['dnn_model.h5']

for name in model_list:
	model = tf.keras.models.load_model(name)
	model.summary()
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	open(name[:-3]+'.tflite', "wb").write(tflite_model)