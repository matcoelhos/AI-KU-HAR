from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
import time
import csv
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

print(tf.__version__)
le = LabelEncoder()

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

violations_list = os.listdir('data')
violations_list.sort()

test_dataset = pd.read_csv('test.csv')
x_test = test_dataset.iloc[:, :-1].values
y_test = le.fit_transform(test_dataset.iloc[:, -1].values)
a,b = x_test.shape[0],x_test.shape[1]
x_test = x_test.tolist()

for i in range(a):
	for j in range(b):
		elem = eval(x_test[i][j])
		x_test[i][j] = elem
x_test = np.array(x_test)
print(x_test.shape)
# Recreate the exact same model, including its weights and the optimizer
dnn_model = tf.keras.models.load_model('dnn_model.h5')
dnn_model.summary()

predictions = []
i = 0
length = len(y_test)
# for i in range(length):
# 	x = x_test[i,:,:]
# 	x = np.reshape(x, (1,x.shape[0],x.shape[1]))
# 	P = dnn_model.predict(x)
# 	pred = np.argmax(P[0])
# 	predictions.append(pred)
# 	print("%d/%d"%(i,length),end='\r')
# 	#input()
# predictions = np.array(predictions)

P = dnn_model.predict(x_test,verbose=True)
predictions = np.argmax(P,axis=-1)
print(predictions)

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
i = 0
for violation in violations_list:
	print('E',violation,sep='')
	i+=1
print(classification_report(y_test, predictions, labels=range(len(violations_list)), target_names=violations_list))
print(confusion_matrix(y_test, predictions))
print('Acc: %.2f %c'%(100*accuracy_score(y_test, predictions),'%'))
print(predictions.shape,y_test.shape)

cfmat = np.array(confusion_matrix(y_test, predictions),dtype='float32')
print(cfmat.shape)

for i in range(cfmat.shape[0]):
    div = np.sum(cfmat[i,:])
    print(i+1)
    for j in range(cfmat.shape[1]):
        if div != 0:
            cfmat[i,j] = cfmat[i,j]/div
        else:
            cfmat[i,j] = 0.

VL = []
for i in range(18):
	VL.append('E%02d'%(i+1))

ax = sns.heatmap(cfmat, annot=True, cmap='Blues',fmt='.1%', vmin= 0, vmax=1)
ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(VL)
ax.yaxis.set_ticklabels(VL)

plt.show()