from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
import numpy as np

import skimage
import csv
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd

class FoodModelConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "foodmodel"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # background + 1 (classes of food)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
config = FoodModelConfig()
config.display()


class InferenceConfig2(FoodModelConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.85
    

inference_config2 = InferenceConfig2()

MODEL_DIR= 'model.h5'

# Recreate the model in inference mode
model2 = modellib.MaskRCNN(mode="inference", 
                          config=inference_config2,
                          model_dir=MODEL_DIR)

model2.load_weights(MODEL_DIR, by_name=True) 

class_names = ['BG', 'Chicken', 'Eba', 'Fish', 'Rice', 'Bread' ]

data = pd.read_csv('food.csv')
data = data.round(3)
X = data.iloc[:,:].values

def call_database(t):
    '''this function gets the values from the data base. 't' is the colon
     you want to print from the data base '''

    py = []
    v = 0
    for x in X[t]:
        tt = '_________'
        py.append('{} ====> {}  {}'.format(data.columns[v],x,tt))
        v=v+1
    return py

def predict(image_paths, img_name):
    img = skimage.io.imread(image_paths)
    img_arr = np.array(img)
    results = model2.detect([img_arr], verbose=1)
    r = results[0]
    pic_names = visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], figsize=(5,5))

    if pic_names == None:
        return None, None
    else:
        if r['class_ids'][0] == 1:
            return call_database(4), pic_names  
        elif r['class_ids'][0] == 2:
            return call_database(3), pic_names
        elif r['class_ids'][0] == 3:
            return call_database(0), pic_names
        elif r['class_ids'][0] == 4:
            return call_database(2), pic_names
        elif r['class_ids'][0] == 5:
            return call_database(1), pic_names  
    
    
    