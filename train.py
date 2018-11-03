# You can put whatever you want in here for training the model,
# provided you document it well enouth for us to understand. Use of
# the argument parser is recommended but not required.

from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Embedding, Input
from keras.callbacks import ModelCheckpoint, CSVLogger

import mycoco
import pickle

import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

mycoco.setmode('train')

all_ids = mycoco.query([['']])

#autoencoder = load_model('./models/autoencoder.model.hdf5')
encoder = load_model('./models/encoder.model.hdf5')
#encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded_flat').output)

# encoder.compile(optimizer='adam', loss='mean_squared_error')

with open('./data/tokenizer10000.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

cap_examples = mycoco.iter_captions_examples(all_ids, tokenizer, encoder)

input_length = 5
embed_size = 50
vocab_size = 10000
img_vec_size = 5000
dropout = 0.2

inputs = Input(shape=(input_length,))
embed = Embedding(vocab_size, embed_size, input_length=input_length)(inputs)
lstm = LSTM(50, dropout=dropout, recurrent_dropout=dropout)(embed)
# Word prediction softmax
word_pred = Dense(vocab_size, activation='softmax', name='word_prediction')(lstm)
image_vec_preds = Dense(img_vec_size, activation = 'sigmoid', name='image_vec_prediction')(lstm)

# This creates a model that includes
# the Input layer and two Dense layers outputs
# model = Model(inputs=inputs, outputs=[word_pred, image_vec_preds])
model = load_model('./assign3.model.hdf5')
model.compile(optimizer='adam',
        loss={
            'word_prediction': 'categorical_crossentropy',
            'image_vec_prediction': 'binary_crossentropy'
        },
        metrics=['accuracy'])
model.summary()

# Logging
csv_logger = CSVLogger('./train_log.csv', append=True, separator=',')
filepath="./model_assign3.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')

# Fit and save
model.fit_generator(cap_examples, steps_per_epoch=2500, epochs=50, verbose=2, callbacks=[checkpoint, csv_logger])
model.save('./assign3.model.hdf5')
