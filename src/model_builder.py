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

    # Freeze the base model layers initially
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    # We explicitly name the base_model layer to find it easily later
    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="ResNet50_TransferLearning")
    return model


def unfreeze_model(model, num_layers_to_unfreeze=20):
    """
    Unfreezes the top N layers of the base ResNet model for Fine-Tuning.
    """
    # 1. Find the ResNet50 layer (it's nested inside our model)
    # We look for the layer that is an instance of the Functional API (which ResNet is)
    base_model = None
    for layer in model.layers:
        if "resnet50" in layer.name:
            base_model = layer
            break

    if base_model is None:
        print("Could not find ResNet50 base layer to unfreeze!")
        return model

    # 2. Set the base model to trainable
    base_model.trainable = True

    # 3. Fine-tune behavior: Freeze all layers EXCEPT the last N layers
    # This prevents destroying the learned weights of the earlier layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    print(f"✅ Unfroze the top {num_layers_to_unfreeze} layers of the ResNet backbone.")
    return model


if __name__ == "__main__":
    model = build_resnet50_model()
    model.summary()