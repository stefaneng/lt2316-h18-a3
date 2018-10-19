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
# Insert whatever other module imports you need.

# Insert whatever initialization code, auxiliary functions, classes,
# global variables, etc, you think you might need, if any.

class PredictiveSearch:
    # Add whatever additional methods etc you need.
    
    def __init__(self, modelfile):
        '''
        Load the model however you want, do whatever initialization you
        need here.  You can change the method signature to require
        more parameters if you think you need them, as long as you document
        this.
        '''
        self.modelfile = modelfile
    
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
        
        raise NotImplementedError #delete when you implement
    
