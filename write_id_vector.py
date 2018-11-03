import mycoco
from keras.models import load_model, Model
import csv
import numpy as np
import pickle
import pandas as pd

mycoco.setmode('train')

autoencoder = load_model('./models/autoencoder.model.hdf5')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded_flat').output)

all_ids = mycoco.query([['']])

all_vectors_iter = mycoco.iter_vector_images(all_ids[0])

max_count = float("inf")
batch = 1000
ids = []
images = []
enc_images = []
j = 0
k = 0
for i, img in all_vectors_iter:
    images.append(img[0])
        #vec = encoder.predict(img) #.round(6)
    ids.append(i)
    #images.append(vec)
    if j >= batch or k >= max_count:
        new_enc = encoder.predict(np.array(images))
        images = []
        enc_images.append(new_enc)
        print("New Batch. Count: {}".format(k))
        #enc_images = np.concatenate([enc_images, new_enc])
        j = 0
        if k >= max_count:
            break
    j += 1
    k += 1
    
new_enc = encoder.predict(np.array(images))
images = []
enc_images.append(new_enc)    
    
enc_images = np.concatenate(enc_images)
with open('./data/enc_images.pickle', 'wb') as f:
    pickle.dump(enc_images, f)    

with open('./data/ids.pickle', 'wb') as f:
    pickle.dump(ids, f)
 
df = pd.DataFrame.from_records(enc_images, index=ids)
df.to_csv('./data/enc_images.csv', float_format='%.4f')