from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
import os
import time
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

print(tf.__version__)
le = LabelEncoder()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


test_dataset = pd.read_csv('test.csv')
x_test = test_dataset.iloc[:, :-1].values
y_test = le.fit_transform(test_dataset.iloc[:, -1].values)
a,b = x_test.shape[0],x_test.shape[1]
x_test = x_test.tolist()

for i in range(a):
	for j in range(b):
		elem = eval(x_test[i][j])
		x_test[i][j] = elem
x_test = np.array(x_test,dtype='float32')
print(x_test.shape)
# Recreate the exact same model, including its weights and the optimizer


model_list = ['dnn_model.tflite']
for model_name in model_list:
	# Recreate the exact same model, including its weights and the optimizer
	interpreter = tf.lite.Interpreter(model_path=model_name)
	interpreter.allocate_tensors()
	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	timefile = open(model_name[:-7]+'.txt','w')

	predictions = []
	i = 0
	length = len(y_test)
	for i in range(length):
		start = time.perf_counter()
		x = x_test[i,:,:]
		x = np.reshape(x, (1,x.shape[0],x.shape[1]))
		interpreter.set_tensor(input_details[0]['index'], x)
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])
		score = int(np.argmax(output_data[0]))
		interval = time.perf_counter() - start
		timefile.write('%.8f\n'%(interval))
		predictions.append(score)
		print("%d/%d"%(i,length),end='\r')
		#input()
	predictions = np.array(predictions)
	print()
	timefile.close()

