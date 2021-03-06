import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data import AUTOTUNE as AUTO



# Directory with data
dir = 'CUB_200_2011'

# Directories for images and segmentation masks
image_dir = os.path.join('..', dir, 'images')
segmentation_dir = os.path.join('..', dir, 'segmentations')

# For initializing models and defining preprocess steps
im_height, im_width = (256, 256)
image_size = (im_height, im_width)
shape = (im_height, im_width, 3)

# For loading data
batch_size = 8
num_parallel_calls = 4
seed = 0

# Set val_split to 0.2501 so you get an even number of samples (8840)
val_split = 0.2501

# Configuration for data generators
datagen_args = dict(validation_split=val_split, 
                    subset="training",
                    image_size=image_size, 
                    batch_size=batch_size,
                    label_mode=None, 
                    shuffle=True, 
                    seed=seed)


# For preparing segmentation masks
def prepare_masks(masks, h=256, w=256):
    masks = tf.slice(masks, [0, 0, 0, 0], [-1, h, w, 1])
    masks = tf.cast(masks != 0, tf.uint8)
    return masks

# For preparing images
def prepare_images(images):
    images = tf.cast(images / 255, tf.float32)
    return images


# Combine images and masks
def create_dataset(img_dir, mask_dir, 
                   img_prep, mask_prep,
                   datagen_args, num_parallel_calls=AUTO):
    """create_training_dataset
    Function for loading images and segmentation masks from specified
    directories, and preprocessing them with specified functions
    Args:
        img_dir, directory for image files
        mask_dir, directory for mask files
        img_prep, function for image preprocessing
        mask_prep, function for mask preprocessing
        datagen_args, args for data generator object
        num_parallel_calls, number of maps to run in parallel

    Returns:
        dataset, a zip object of our preprocessed images/masks
    """

    # Load image data
    print("Loading image data...")
    image_ds = image_dataset_from_directory(img_dir, **datagen_args)
    image_ds = image_ds.map(img_prep, num_parallel_calls=num_parallel_calls)

    # Load segmentation data
    print("\nLoading segmentation data...")
    mask_ds = image_dataset_from_directory(mask_dir, **datagen_args)
    mask_ds = mask_ds.map(mask_prep, num_parallel_calls=num_parallel_calls)
    
    return image_ds, mask_ds
