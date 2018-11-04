# You can put whatever you want in here for training the model,
# provided you document it well enouth for us to understand. Use of
# the argument parser is recommended but not required.

from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Embedding, Input, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam

import mycoco
import pickle

mycoco.setmode('train')

# Get all the image ids
all_ids = mycoco.query([['']])

# Load the autoencoder model
autoencoder = load_model('./models/autoencoder.model.hdf5')
# Extract the flat encoding from the autoencoder model in layer `encoded_flat`
# Outputs vectors of length 200
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded_flat').output)

# Load the tokenizer used to train the language model
# Reused from assignment 2
with open('./data/tokenizer10000.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Iterator that returns a random caption with an encoded image
cap_examples = mycoco.iter_captions_examples(all_ids, tokenizer, encoder)

input_length = 5
embed_size = 50
vocab_size = 10000
img_vec_size = autoencoder.get_layer('encoded_flat').output_shape[1]
dropout = 0.2

inputs = Input(shape=(input_length,))
embed = Embedding(vocab_size, embed_size, input_length=input_length)(inputs)
lstm = LSTM(50, dropout=dropout, recurrent_dropout=dropout)(embed)
# Word prediction softmax
word_pred = Dense(128, activation='relu')(lstm)
word_pred = Dense(vocab_size, activation='softmax', name='word_prediction')(word_pred)

image_vec_preds = Dropout(dropout)(lstm)
image_vec_preds = Dense(128, activation='relu')(image_vec_preds)
image_vec_preds = Dropout(dropout)(image_vec_preds)
image_vec_preds = Dense(64, activation='relu')(image_vec_preds)
image_vec_preds = Dense(img_vec_size, name='image_vec_prediction')(image_vec_preds)

model = Model(inputs=inputs, outputs=[word_pred, image_vec_preds])
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,
        loss={
            'word_prediction': 'categorical_crossentropy',
            'image_vec_prediction': 'mean_squared_error'
        },
        metrics=['accuracy'])
model.summary()

# Logging
csv_logger = CSVLogger('./train_log.csv', append=True, separator=',')
filepath="./model_assign3.checkpoint.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=2)

# Fit and save
model.fit_generator(cap_examples, steps_per_epoch=1000, epochs=500, verbose=2, callbacks=[checkpoint, csv_logger])
model.save('./models/assign3.model.hdf5')
