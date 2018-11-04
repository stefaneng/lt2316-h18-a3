import mycoco
from keras.models import load_model, Model
import csv
import numpy as np
import pickle
import pandas as pd

mycoco.setmode('train')

# Load the models
autoencoder = load_model('./models/autoencoder.model.hdf5')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded_flat').output)

# Get all the image ids from training set
all_ids = mycoco.query([['']])

all_vectors_iter = mycoco.iter_vector_images(all_ids[0])

# Predict a batch of images at a time
# This prevents memory overflow since loading all the images is huge.
max_count = float("inf")
batch = 1000
ids = []
images = []
enc_images = []
j = 0
k = 0
for i, img in all_vectors_iter:
    images.append(img[0])
    ids.append(i)
    if j >= batch or k >= max_count:
        new_enc = encoder.predict(np.array(images))
        images = []
        enc_images.append(new_enc)
        j = 0
        if k >= max_count:
            break
    j += 1
    k += 1

new_enc = encoder.predict(np.array(images))
images = []
enc_images.append(new_enc)

enc_images = np.concatenate(enc_images)

# Create a dataframe and write to csv with 4 decimal places
df = pd.DataFrame.from_records(enc_images, index=ids)
df.to_csv('./data/enc_images.csv', float_format='%.4f')
