import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = r"C:\_Projects\home-energy-ai\data\jena_climate_2009_2016_extracted\jena_climate_2009_2016.csv"

df = pd.read_csv(csv_path)
# print(df)

df = df[5::6]
# print(df)


df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
# print(df[:26])

temp = df['T (degC)']


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

WINDOW_SIZE = 5
X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
X1.shape, y1.shape


X_train1, y_train1 = X1[:60000], y1[:60000]
X_val1, y_val1 = X1[60000:65000], y1[60000:65000]



from tensorflow.keras.models import load_model
model1 = load_model('old/models/model1/model.keras')




train_predictions = model1.predict(X_train1).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})



plt.plot(train_results['Train Predictions'][50:100])
plt.plot(train_results['Actuals'][50:100])
plt.show()



val_predictions = model1.predict(X_val1).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
print(val_results)
