# This is a helper module that contains conveniences to access the MS COCO
# dataset. You can modify at will.  In fact, you will almost certainly have
# to, or implement otherwise.

import sys

# This is evil, forgive me, but practical under the circumstances.
# It's a hardcoded access to the COCO API.  
COCOAPI_PATH='/scratch/lt2316-h18-resources/cocoapi/PythonAPI/'
TRAIN_ANNOT_FILE='/scratch/lt2316-h18-resources/coco/annotations/instances_train2017.json'
VAL_ANNOT_FILE='/scratch/lt2316-h18-resources/coco/annotations/instances_val2017.json'
TRAIN_CAP_FILE='/scratch/lt2316-h18-resources/coco/annotations/captions_train2017.json'
VAL_CAP_FILE='/scratch/lt2316-h18-resources/coco/annotations/captions_val2017.json'
TRAIN_IMG_DIR='/scratch/lt2316-h18-resources/coco/train2017/'
VAL_IMG_DIR='/scratch/lt2316-h18-resources/coco/val2017/'
annotfile = TRAIN_ANNOT_FILE
capfile = TRAIN_CAP_FILE
imgdir = TRAIN_IMG_DIR

sys.path.append(COCOAPI_PATH)
from pycocotools.coco import COCO

annotcoco = None
capcoco = None
catdict = {}

# OK back to normal.
import random
import skimage.io as io
import skimage.transform as tform
import numpy as np

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# See: https://github.com/keras-team/keras/issues/6462
import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

def setmode(mode):
    '''
    Set entire module's mode as 'train' or 'test' for the purpose of data extraction.
    '''
    global annotfile
    global capfile
    global imgdir
    global annotcoco, capcoco
    global catdict
    if mode == "train":
        annotfile = TRAIN_ANNOT_FILE
        capfile = TRAIN_CAP_FILE
        imgdir = TRAIN_IMG_DIR
    elif mode == "test":
        annotfile = VAL_ANNOT_FILE
        capfile = VAL_CAP_FILE
        imgdir = VAL_IMG_DIR
    else:
        raise ValueError

    annotcoco = COCO(annotfile)
    capcoco = COCO(capfile)

    # To facilitate category lookup.
    cats = annotcoco.getCatIds()
    catdict = {x:(annotcoco.loadCats(ids=[x])[0]['name']) for x in cats}
    
def query(queries, exclusive=True):
    '''
    Collects mutually-exclusive lists of COCO ids by queries, so returns 
    a parallel list of lists.
    (Setting 'exclusive' to False makes the lists non-exclusive.)  
    e.g., exclusive_query([['toilet', 'boat'], ['umbrella', 'bench']])
    to find two mutually exclusive lists of images, one with toilets and
    boats, and the other with umbrellas and benches in the same image.
    '''
    if not annotcoco:
        raise ValueError
    imgsets = [set(annotcoco.getImgIds(catIds=annotcoco.getCatIds(catNms=x))) for x in queries]
    if len(queries) > 1:
        if exclusive:
            common = set.intersection(*imgsets)
            return [[x for x in y if x not in common] for y in imgsets]
        else:
            return [list(y) for y in imgsets]
    else:
        return [list(imgsets[0])]
    
def get_captions_for_ids(idlist):
    annids =  capcoco.getAnnIds(imgIds=idlist)
    anns = capcoco.loadAnns(annids)
    return [ann['caption'] for ann in anns]    

def get_cats_for_img(imgid):
    '''
    Takes an image id and gets a category list for it.
    '''
    if not annotcoco:
        raise ValueError
        
    imgannids = annotcoco.getAnnIds(imgIds=imgid)
    imganns = annotcoco.loadAnns(imgannids)
    return list(set([catdict[x['category_id']] for x in imganns]))
    
    
def iter_captions(idlists, cats, batch=1):
    '''
    Obtains the corresponding captions from multiple COCO id lists.
    Randomizes the order.  
    Returns an infinite iterator (do not convert to list!) that returns tuples (captions, categories)
    as parallel lists at size of batch.
    '''
    if not capcoco:
        raise ValueError
    if batch < 1:
        raise ValueError

    full = []
    for z in zip(idlists, cats):
        for x in z[0]:
            full.append((x, z[1]))
        
    while True:
        randomlist = random.sample(full, k=len(full))
        captions = []
        labels = []

        for p in randomlist:
            annids =  capcoco.getAnnIds(imgIds=[p[0]])
            anns = capcoco.loadAnns(annids)
            for ann in anns:
                captions.append(ann['caption'])
                # For LSTM you may want to do more with the captions
                # or otherwise distribute the data.
                labels.append(p[1])
                if len(captions) % batch == 0:
                    yield (captions, labels)
                    captions = []
                    labels = []
                    
def iter_captions_examples(idlists, tokenizer, model, num_words = 10000, seq_maxlen = 5, batch=1):
    '''
    Obtains the corresponding caption training examples from multiple COCO id lists.
    Randomizes the order.  
    Returns an infinite iterator (do not convert to list!) that returns tuples (captions, categories)
    as parallel lists at size of batch.
    '''
    if not capcoco:
        raise ValueError
    if batch < 1:
        raise ValueError

    full = []
    for z in idlists:
        for x in z:
            full.append(x)
        
    while True:
        randomlist = random.sample(full, k=len(full))
        caption_examples = []
        # encoded_image = []

        for p in randomlist:
            # Get the captions
            annids =  capcoco.getAnnIds(imgIds=[p])
            anns = capcoco.loadAnns(annids)
            
            # Get the image
            size = (200,200)
            imgfile = annotcoco.loadImgs([p])[0]['file_name']
            img = io.imread(imgdir + imgfile)
            imgscaled = tform.resize(img, size)
            
            # Colour images only.
            if imgscaled.shape != (size[0], size[1], 3):
                continue
                        
            # See https://github.com/keras-team/keras/issues/6462
            with graph.as_default():
                # Use model to get the encoded image
                encoded_img = model.predict(np.array([imgscaled]))
                
            for ann in anns:                
                cap = ann['caption']
                # print("Caption:", cap)
                encoded = tokenizer.texts_to_sequences([cap])[0]
                # print("Encoded:", encoded)
                # Create 
                for i in range(1,len(encoded)):
                    end_index = len(encoded) - i
                    # Force the sequence to fit into seq_maxlen
                    start_index = end_index - seq_maxlen
                    if start_index < 0:
                        start_index = 0            
                    cap_ex = encoded[start_index:end_index]
                    pad_cap_ex = pad_sequences([cap_ex], padding='post', maxlen=seq_maxlen)
                    pred_word = encoded[-i]
                    y_words = to_categorical(pred_word, num_classes=num_words)
                    
                    # This batch this isn't really going to work in the way
                    # Would need to keep track of multiple, for now just run with batch=1
#                    if len(caption_examples) % batch == 0:
                    yield (pad_cap_ex, [[y_words], encoded_img])
                # For LSTM you may want to do more with the captions
                # or otherwise distribute the data.

                    
def iter_captions_cats(idlists, cats, batch=1):
    '''
    Obtains the corresponding captions from multiple COCO id lists alongside all associated image captions per image.
    Randomizes the order.  
    Returns an infinite iterator (do not convert to list!) that returns tuples (captions, categories)
    as parallel lists at size of batch.
    '''
    if not capcoco:
        raise ValueError
    if batch < 1:
        raise ValueError

    full = []
    for z in zip(idlists, cats):
        for x in z[0]:
            full.append((x, z[1]))
        
    while True:
        randomlist = random.sample(full, k=len(full))
        captions = []
        labels = []

        for p in randomlist:
            annids =  capcoco.getAnnIds(imgIds=[p[0]])
            anns = capcoco.loadAnns(annids)
            for ann in anns:
                imgid = ann['image_id']
                cats = get_cats_for_img(imgid)
                captions.append((ann['caption'], cats))
                # For LSTM you may want to do more with the captions
                # or otherwise distribute the data.
                labels.append(p[1])
                if len(captions) % batch == 0:
                    yield (captions, labels)
                    captions = []
                    labels = []
                    
def iter_images(idlists, cats, size=(200,200), batch=1):
    '''
    Obtains the corresponding image data as numpy array from multiple COCO id lists.
    Returns an infinite iterator (do not convert to list!) that returns tuples (imagess, categories)
    as parallel lists at size of batch.
    By default, randomizes the order and resizes the image.
    '''
    if not annotcoco:
        raise ValueError
    if batch < 1:
        raise ValueError
    if not size:
        raise ValueError # size is mandatory

    full = []
    for z in zip(idlists, cats):
        for x in z[0]:
            full.append((x, z[1]))

    while True:
        randomlist = random.sample(full, k=len(full))

        images = []
        labels = []
        for r in randomlist:
            imgfile = annotcoco.loadImgs([r[0]])[0]['file_name']
            img = io.imread(imgdir + imgfile)
            imgscaled = tform.resize(img, size)
            # Colour images only.
            if imgscaled.shape == (size[0], size[1], 3):
                images.append(imgscaled)
                labels.append(r[1])
                if len(images) % batch == 0:
                    yield (np.array(images), np.array(labels))
                    images = []
                    labels = []
