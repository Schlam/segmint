import os
import numpy as np

# Enable XLA for faster training 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf
from tensorflow.data import AUTOTUNE as AUTO

# display tensorflow version to stdout
print("tensorflow version:", tf.__version__)

# packages within this repository
from utils.data import create_dataset, datagen_args
from utils.data import prepare_images, prepare_masks
from models import build_unet


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
DATA LOADING

configure data loading parameters below

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""


# Directory with data
DIR = 'CUB_200_2011'

# Directories for images and segmentation masks
IMAGE_DIR = os.path.join('..', DIR, 'images')
MASK_DIR = os.path.join('..', DIR, 'segmentations')

# For initializing models and defining preprocess steps
SHAPE = (256, 256, 3)


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
DATA PREPARATION

configure data preparation parameters below

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""


# For loading/preparing data
BATCH_SIZE = 4
NUM_PARALLEL = 4



"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
TRAINING

configure training parameters below

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""

# For training
EPOCHS = 1
TOTAL_NUM_TRAIN = 8840

# For model compiling
OPTIMIZER = "adam"
METRICS = ["accuracy"]
LOSS = "categorical_crossentropy"

# For model checkpoints
CHKPT = "model.val_loss-{val_loss:.2f}_epoch-{epoch:d}.h5"


# Callbacks
CALLBACKS = [tf.keras.callbacks.ModelCheckpoint(filepath=CHKPT),
             tf.keras.callbacks.ReduceLROnPlateau(),
             tf.keras.callbacks.EarlyStopping()]


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
RUN TRAINING REGIMEN

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""

def run():

    # Load image data
    print("Loading image data...")
    image_ds = image_dataset_from_directory(img_dir, **datagen_args)
    image_ds = image_ds.map(img_prep, num_parallel_calls=num_parallel_calls)

    # Load segmentation data
    print("\nLoading segmentation data...")
    mask_ds = image_dataset_from_directory(mask_dir, **datagen_args)
    mask_ds = mask_ds.map(mask_prep, num_parallel_calls=num_parallel_calls)


    # Load images and masks into their own separate datasets
    images, masks = create_dataset(IMAGE_DIR, MASK_DIR,
                                   prepare_images, prepare_masks,
                                   datagen_args, num_parallel_calls=NUM_PARALLEL)

    
    # Create model
    inputs = tf.keras.Input(SHAPE)
    model = build_unet(inputs, n_classes=1)
    model.compile(loss=LOSS,
                  optimizer=OPTIMIZER,
                  metrics=METRICS)

    # Fit model
    history = model.fit(
        zip(images, masks),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=(TOTAL_NUM_TRAIN // BATCH_SIZE),
        callbacks=CALLBACKS)



# Combine images and masks
def create_dataset(img_dir, mask_dir,
                   img_prep, mask_prep,
                   datagen_args, num_parallel_calls=AUTO):    

if __name__ == "__main__":

    run()
