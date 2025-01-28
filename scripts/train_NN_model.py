import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True,
#     cache_dir='.',      # Current directory
#     cache_subdir='data') # Put everything under ./data)


csv_path = r"C:\_Projects\home-energy-ai\data\jena_climate_2009_2016_extracted\jena_climate_2009_2016.csv"

df = pd.read_csv(csv_path)
# print(df)

df = df[5::6]
# print(df)


df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
# print(df[:26])

temp = df['T (degC)']
temp.plot()
plt.show()


def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)


print("IM HERE 1")

WINDOW_SIZE = 5
X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
X1.shape, y1.shape

print("IM HERE 2")

X_train1, y_train1 = X1[:60000], y1[:60000]
X_val1, y_val1 = X1[60000:65000], y1[60000:65000]
X_test1, y_test1 = X1[65000:], y1[65000:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape

print("IM HERE 3")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model1 = Sequential()
model1.add(InputLayer((7, 1)))
model1.add(LSTM(1028))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

print("IM HERE 4")
model1.summary()

print("IM HERE 5")

cp1 = ModelCheckpoint('models/model1/model.keras', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

print("IM HERE 6")

model1.fit(X_train1, y_train1, 
        validation_data=(X_val1, y_val1),
        epochs=10, 
        callbacks=[cp1])

print("DONE TRAINING")


