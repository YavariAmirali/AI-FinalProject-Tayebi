import tensorflow as tf


def tf_preprocess_image(image, label):

    # 1. Ensure image is float32 (Required for neural networks)
    image = tf.cast(image, tf.float32)

    # 2. Normalize pixel values to be between 0 and 1
    image = image / 255.0

    # Note: Resizing is already handled by image_dataset_from_directory in train.py
    return image, label