import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from  keras.layers import Input
import pandas as pd
import io
from math import log
import numpy as np
from sklearn.preprocessing import LabelEncoder

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(tf.__version__)
le = LabelEncoder()

train_dataset = pd.read_csv('train.csv')
x_train = train_dataset.iloc[:, :-1].values
y_train = le.fit_transform(train_dataset.iloc[:, -1].values)
print(y_train.shape)
a,b = x_train.shape[0],x_train.shape[1]
x_train = x_train.tolist()

for i in range(a):
	for j in range(b):
		elem = eval(x_train[i][j])
		if i == 0 and j == 0:
			num_features = len(elem)
		x_train[i][j] = elem
x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],num_features))
print(x_train.shape)

validation_dataset = pd.read_csv('validation.csv')
x_val = validation_dataset.iloc[:, :-1].values
y_val = le.fit_transform(validation_dataset.iloc[:, -1].values)
print(y_val.shape)
a,b = x_val.shape[0],x_val.shape[1]
x_val = x_val.tolist()

for i in range(a):
	for j in range(b):
		elem = eval(x_val[i][j])
		x_val[i][j] = elem
x_val = np.array(x_val)
x_val = np.reshape(x_val, (x_val.shape[0],x_val.shape[1],num_features))
print(x_val.shape)

factor = 1e-4

il = layers.Input(shape=(x_val.shape[1],num_features))
x = layers.LSTM(128,return_sequences=True,kernel_regularizer=l2(factor))(il)
x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.2)(x)
x = layers.Dense(256,kernel_regularizer=l2(factor))(x)
x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.2)(x)
out = layers.Dense(18,kernel_regularizer=l2(factor), activation='softmax')(x)

model = tf.keras.Model(inputs=il,outputs=[out])

# model = tf.keras.models.Sequential([
# 	layers.LSTM(128,return_sequences=True),
# 	# layers.LSTM(40,return_sequences=True,go_backwards=True),
# 	layers.Flatten(),
# 	layers.BatchNormalization(),
# 	layers.Dense(512),
# 	layers.BatchNormalization(),
# 	layers.Dense(18, activation='softmax')
# 	])

# model = tf.keras.models.Sequential([
# 	layers.Flatten(),
# 	layers.BatchNormalization(),
# 	# layers.Dropout(0.2),
# 	layers.Dense(256, activation='relu',input_shape= (x_train.shape[1],)),
# 	layers.BatchNormalization(),
# 	# layers.Dropout(0.2),
# 	layers.Dense(256, activation='relu'),
# 	layers.BatchNormalization(),
# 	# layers.Dropout(0.2),
# 	layers.Dense(256, activation='relu'),
# 	layers.BatchNormalization(),
# 	# layers.Dropout(0.2),
# 	layers.Dense(18, activation='softmax')
# 	])

class_weights = compute_class_weight('balanced',classes=np.unique(y_train),y=y_train)
for i in range(len(class_weights)):
	class_weights[i] = 1
	# class_weights[i] = min(class_weights[i],1.2)
	# class_weights[i] = max(class_weights[i],0.8)

class_weights[-4] = 4
class_weights[2] = 4
class_weights[5] = 1.2
# class_weights[-7] = 0.7
class_weights[-5] = 2

print(class_weights)
class_weight_dict = dict(enumerate(class_weights))

num_epochs = 30000

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

# model.load_weights('dnn_model.h5')

model.compile(optimizer=optimizer,
				loss='sparse_categorical_crossentropy',
				metrics = ['accuracy'])


reduceonplateu = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=3,
    min_delta=0.0001,
)

ealystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=15,
    restore_best_weights=True
)

history = model.fit(x_train,y_train,
					validation_data=(x_val,y_val),
					batch_size=16,
					epochs = num_epochs,
					steps_per_epoch = 1000,
					# validation_steps = 600,
					verbose = 1,
					class_weight=class_weight_dict,
					use_multiprocessing=True,
					workers=4,
					callbacks=[reduceonplateu,ealystopping]
	)
model.save('dnn_model.h5')

model.evaluate(x_val,y_val)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()