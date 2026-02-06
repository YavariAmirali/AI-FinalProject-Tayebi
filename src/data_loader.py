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

        # Automatically detect class names from the folder structure
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
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])

        self.indices = np.arange(len(self.image_paths))


        self.augmentation = A.Compose(
            [
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ZoomBlur(max_factor=1.1, p=0.2),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
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
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        images = []
        labels = []

        for i in batch_indices:
            file_path = self.image_paths[i]
            label = self.labels[i]

            img = cv2.imread(file_path)

            if img is None:
                continue
            # OpenCV loads as BGR by default, so we must switch to RGB for the model
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.augment:
                img = self.augmentation(image=img)["image"]
            else:
                img = cv2.resize(img, self.target_size)

            # Preprocess inputs specifically for ResNet50 (zero-centering
            img = preprocess_input(img)

            images.append(img)
            labels.append(label)

        return np.array(images), np.array(labels)


def get_data_loaders(data_base_path, batch_size=32):
    # Train loader gets augmentation and shuffling turned on
    train_loader = MedicalDataGenerator(
        os.path.join(data_base_path, "train"),
        batch_size=batch_size,
        augment=True,
        shuffle=True,
    )
    # Validation loader should provide clean, un-augmented images
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



def main():

    train_gen, val_gen, test_gen = get_data_loaders(DATA_DIR, batch_size=8)

    print("Train batches:", len(train_gen))
    print("Val batches:", len(val_gen))
    print("Test batches:", len(test_gen))


    x, y = train_gen[0]

    print("Images shape:", x.shape)
    print("Labels shape:", y.shape)
    print("Labels:", y)


    print("Pixel min:", x.min())
    print("Pixel max:", x.max())


    xv, yv = val_gen[0]
    print("Val batch shape:", xv.shape)


    xt, yt = test_gen[0]
    print("Test batch shape:", xt.shape)

 
    print("Class mapping:", train_gen.class_to_idx)

    print("✅ Data loader sanity test passed!")


if __name__ == "__main__":
    main()