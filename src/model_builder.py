import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_resnet50_model(input_shape=(224, 224, 3)):
    """
    Builds a ResNet50 model with transfer learning from ImageNet.
    The base layers are frozen, and a custom head is added for binary classification.
    """

    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the base model layers
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    # Pass inputs through the pre-trained base model
    x = base_model(inputs, training=False)

    # Add custom classification head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer for binary classification (Pneumonia vs Normal)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="ResNet50_TransferLearning")

    return model


if __name__ == "__main__":
    model = build_resnet50_model()
    model.summary()
    print("Model built successfully.")