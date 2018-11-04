# This file is the primary way that your assignment will be tested.
# It will be loaded as a Jupyter notebook, and the "predictive_search"
# function will be called in the notebook line.
#
# You can add whatever functions you want, as long as the PredictiveSearch
# class is present in this file.  You can import whatever you need,
# both your own modules, mycoco.py, and third party modules available on mltgpu
# (other than what literally solves the assignment, if any such thing
# exists).

import mycoco
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

import numpy as np
import pandas as pd
# Insert whatever other module imports you need.

# Insert whatever initialization code, auxiliary functions, classes,
# global variables, etc, you think you might need, if any.

class PredictiveSearch:
    # Add whatever additional methods etc you need.
    
    def __init__(self, modelfile, tokenizerfile, encoded_images, test=False):
        '''
        Load the model however you want, do whatever initialization you
        need here.  You can change the method signature to require
        more parameters if you think you need them, as long as you document
        this.
        `tokenizerfile` path to the pickled tokenizer used in the model.
        `encoded_images` path to csv file with image ids and encoded vectors.
        '''
        self.model = load_model(modelfile)
        #self.model.summary()
        # read in the tokenizer,
        with open(tokenizerfile, 'rb') as f:
            self.tokenizer = pickle.load(f)
            
        mycoco.setmode('train')

        # Load in the pre-computed vectors for each id in training set
        self.encoded_df = pd.read_csv(encoded_images, compression='gzip', index_col=0).dropna(axis=0)
        
    def nearest_images(self, k, encoded_vec):
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.encoded_df)        
        self.encoded_df.index[nbrs.kneighbors([encoded_vec], return_distance=False)[0]]
        
        # Get the k nearest neighbors to the encoded vector
        dists, nearest_indicies = nbrs.kneighbors([encoded_vec])
        # Return the images ids
        nearest_indicies = nearest_indicies[0]
        # print("Nearest indicies:", nearest_indicies) 
        # print("Distances:", dists)
        nearest_img_ids = self.encoded_df.index[nearest_indicies]
        # Show the top k images
        for img in mycoco.get_images(nearest_img_ids):
            plt.axis('off')
            plt.imshow(img)
            plt.show()
    
    def predictive_search(self, words):
        '''
        Based on the loaded model data, do the following:
        
        0. Process the text so that it is compatible with your model input.
           (I.e., do whatever tokenization you did for training, etc.)
        1. Strip out all the out-of-vocabulary words (ie, words not
           represented in the model). Print them in a message/list of 
           messages. 
        2. If there are no words left, raise an error.
        3. From the remaining words, predict and print out the top five
           most likely words according to the model to be the next word.
        4. Predict and display (using e.g. plt.show(), so that it appears
           on the Jupyter console) the top three corresponding images
           that the model predicts.
        '''
        
        # Show top 5 predictions
        predicted_number = 5
        vocab_size = self.tokenizer.num_words
        words_filtered = text_to_word_sequence(words)
        # Flip the word index around so we can look up word names based on the index
        word_lookup = {index: w for w, index in self.tokenizer.word_index.items() if index < vocab_size}
        
        print("Original sentence:", words)      
        
        # Print words that are out of vocabulary range
        for w in words_filtered:            
            if w not in self.tokenizer.word_index or self.tokenizer.word_index[w] >= vocab_size:
                print(w, "missing from tokenizer")       

        # Get the window size from the model input
        window_size = self.model.layers[0].get_input_at(0).get_shape().as_list()[1]
        print("Window size =", window_size)

        encoded = self.tokenizer.texts_to_sequences([words])   
        
        if len(encoded[0]) == 0:
            raise RuntimeError('No words from the sentence are in the vocabulary') 
        
        x = pad_sequences(encoded, padding='post', truncating='pre', maxlen=window_size)
        
        print("Predicting using words: ", " ".join([word_lookup[j] for j in x[0] if j != 0]))

        word_preds, vec_preds = self.model.predict(np.array(x))
        
        # Only are predicting one value
        words_preds = word_preds[0]
        vec_preds = vec_preds[0]

        # Use this prediction as the last word
        pred = np.argmax(word_preds, axis=None) + 1
        
        # Word predictions
        # Descending order
        sort_word_preds = np.argsort(word_preds, axis=None)[::-1][:predicted_number]
        sort_word_names = [word_lookup[i + 1] for i in sort_word_preds]
        sort_word_probs = words_preds[sort_word_preds]

        print("Predicting: {}...".format(words))
        print("Word Predictions:")
        for w, prob in list(zip(sort_word_names, sort_word_probs)):
            print("\t{}: {}".format(w, prob))
            
        # print("Vector prediction:", vec_preds.round(3))
        self.nearest_images(3, vec_preds)
