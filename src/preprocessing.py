import tensorflow as tf

def tf_preprocess_image(image, label):
    """
    Converts image to float32 and normalizes to [0, 1].
    Expected input: image tensor from dataset, label.
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def augment_data(image, label):
    """
    Applies random augmentation to training data.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label
