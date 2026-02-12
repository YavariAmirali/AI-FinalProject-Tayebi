import tensorflow as tf
import numpy as np
import albumentations as A
import cv2
import os
from tensorflow.keras.applications.resnet50 import preprocess_input


class MedicalDataGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            directory,
            batch_size=32,
            target_size=(224, 224),
            augment=False,
            shuffle=True,
    ):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle

        self.image_paths = []
        self.labels = []

        self.class_names = sorted(
            [
                d
                for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))
            ]
        )

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = os.path.join(directory, class_name)
            # Sort filenames! Essential for matching errors to files
            for img_name in sorted(os.listdir(class_dir)):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])

        self.indices = np.arange(len(self.image_paths))

        # Medical Augmentations
        self.augmentation = A.Compose(
            [
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ZoomBlur(max_factor=1.1, p=0.2),
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.Resize(target_size[0], target_size[1]),
            ]
        )

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[
                        idx * self.batch_size: (idx + 1) * self.batch_size
                        ]

        images = []
        batch_labels = []

        for i in batch_indices:
            file_path = self.image_paths[i]
            label = self.labels[i]

            img = cv2.imread(file_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.augment:
                img = self.augmentation(image=img)["image"]
            else:
                img = cv2.resize(img, self.target_size)

            # ResNet Specific Preprocessing
            img = preprocess_input(img)

            images.append(img)
            batch_labels.append(label)

        return np.array(images), np.array(batch_labels)

    def get_labels(self):
        return self.labels

    def get_filepaths(self):
        return self.image_paths


def get_data_loaders(data_base_path, batch_size=32):
    train_loader = MedicalDataGenerator(
        os.path.join(data_base_path, "train"),
        batch_size=batch_size,
        augment=True,
        shuffle=True,
    )

    val_loader = MedicalDataGenerator(
        os.path.join(data_base_path, "val"),
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )

    test_loader = MedicalDataGenerator(
        os.path.join(data_base_path, "test"),
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)

DATA_DIR = os.path.join(project_root, "data")