import tensorflow as tf
import os
import pandas as pd
import numpy as np



# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True,
#     cache_dir='.',      # Current directory
#     cache_subdir='data') # Put everything under ./data)


csv_path = "C:\_Projects\home-energy-ai\data\jena_climate_2009_2016_extracted\jena_climate_2009_2016.csv"

df = pd.read_csv(csv_path)
# print(df)

df = df[5::6]
# print(df)


df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
# print(df[:26])

temp = df['T (degC)']
temp.plot()
