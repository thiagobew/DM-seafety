import seaborn as sns
import zipfile
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator, DataFrameIterator
import os
from sklearn.metrics import confusion_matrix
from os.path import exists
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# MIGHT BE USEFUL


def extract_zip_folder(zip_name: str) -> None:
    """Extracts a zip folder to a given path.

    Args:
        zip_folder_path (str): name of the zip folder.
    """
    ASSETS_PATH = os.path.join(os.getcwd(), 'dataset')
    zip_path = os.path.join(ASSETS_PATH, zip_name)
    zip = zipfile.ZipFile(file=zip_path, mode='r')
    zip.extractall(os.path.join(ASSETS_PATH))
    zip.close()

# IF IT DOESN'T WORK, TRY CHANGING CONVS LAYERS OR PREPROPCESSING.

class CaddyClassifier:
    def __init__(self, image_shape: tuple[int, int, int], batch_size: int = 32, re_train: bool = False) -> None:
        self._image_shape = image_shape
        self._batch_size = batch_size
        self._model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model')
        self._dataset_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dataset')
        self._training_dataset, self._validation_dataset, self._test_dataset = self._load_datasets()
        self._model: Sequential = self._load(re_train=re_train)

    def _load_datasets(self) -> tuple[DataFrameIterator, DataFrameIterator, DataFrameIterator]:
        images, labels = self._get_images_and_labels(self._dataset_path)

        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42)

        training_dataset, validation_dataset = self._get_train_val_datasets(
            train_images, train_labels)
        test_dataset = self._get_test_dataset(test_images, test_labels)

        return training_dataset, validation_dataset, test_dataset

    def _get_images_and_labels(self, dataset_path: str) -> tuple[list[str], list[str]]:
        image_filenames = os.listdir(dataset_path)
        images = [os.path.join(dataset_path, filename)
                  for filename in image_filenames]
        # Assuming filename format: "label_image.jpg"
        labels = [filename.split("_")[0] for filename in image_filenames]

        classes_folders = os.listdir(dataset_path)
        images = []
        labels = []
        for folder in classes_folders:
            folder_path = os.path.join(dataset_path, folder)
            for filename in os.listdir(folder_path):
                images.append(os.path.join(folder_path, filename))
                labels.append(folder),

        return images, labels

    def _get_train_val_datasets(self, images: list[str], labels: list[str]) -> tuple[DataFrameIterator, DataFrameIterator]:
        train_datagen = ImageDataGenerator(
            rescale=1./255, validation_split=0.2)  # set validation split
        x, y, z = self._image_shape

        train_dataset = train_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({"filename": images, "label": labels}),
            x_col="filename",
            y_col="label",
            target_size=(x, y),
            batch_size=self._batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='training')  # set as training data

        validation_dataset = train_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({"filename": images, "label": labels}),
            x_col="filename",
            y_col="label",
            target_size=(x, y),
            batch_size=self._batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='validation')  # set as validation data

        return train_dataset, validation_dataset

    def _get_test_dataset(self, images: list[str], labels: list[str]) -> DataFrameIterator:
        generator = ImageDataGenerator(rescale=1./255)
        x, y, z = self._image_shape
        dataset = generator.flow_from_dataframe(dataframe=pd.DataFrame({"filename": images, "label": labels}),
                                                x_col="filename",
                                                y_col="label",
                                                target_size=(x, y),
                                                batch_size=self._batch_size,
                                                class_mode='categorical',
                                                shuffle=True)

        return dataset

    def _load(self, re_train: bool) -> Sequential:
        if exists(os.path.join(self._model_path, 'caddy_model.h5')):
            if re_train:
                classifier = self._train()
            else:
                classifier = load_model(os.path.join(
                    self._model_path, 'caddy_model.h5'))
        else:
            classifier = self._train()

        return classifier

    def _train(self) -> Sequential:
        classifier = self._compile_model()

        # Trains the classifier
        classifier.fit(self._training_dataset, validation_data=self._validation_dataset,
                       callbacks=[EarlyStopping(
                           monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)],
                       epochs=50)

        # Saves the classifier
        classifier.save(os.path.join(self._model_path, "caddy_model.h5"))

        return classifier

    def _compile_model(self) -> Sequential:
        # Create a sequential model
        model = Sequential(name='CaddyClassifier')

        # Input layer
        model.add(Input(shape=self._image_shape))

        # Convolutional layers
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        # Flatten layer
        model.add(Flatten())
        units = model.output_shape[1]

        # Dense layers
        model.add(Dense(units=units, activation='relu'))
        
        units = model.output_shape[1] // 2
        
        model.add(Dense(units=units, activation='relu'))
        
        units = model.output_shape[1] // 2
        
        model.add(Dense(units=units, activation='relu'))
        
        units = model.output_shape[1] // 2
        
        model.add(Dense(units=units, activation='relu'))
        
        units = model.output_shape[1] // 2
        
        model.add(Dense(units=units, activation='relu'))
        
        units = model.output_shape[1] // 2
        
        model.add(Dense(units=units, activation='relu'))

        # Output layer
        model.add(Dense(units=16, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        return model

    def predict(self, dataset_path: str) -> str:
        images, labels = self._get_images_and_labels(dataset_path)
        dataset = self._get_dataset(images, labels)

        predictions = self._model.predict(dataset)
        predictions = np.argmax(predictions, axis=1)

        return predictions

    def evaluate(self) -> np.ndarray:
        predictions = self._model.predict(self._test_dataset)
        predictions = np.argmax(predictions, axis=1)

        labels = self._test_dataset.classes

        cm = confusion_matrix(labels, predictions)

        # Adjust the figure size as per your preference
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

        # Add labels, title, and other formatting
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        return cm

    def test(self) -> np.ndarray:
        predictions = self._model.predict(self._test_dataset)
        predictions = np.argmax(predictions, axis=1)

        labels = self._test_dataset.classes

        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True)

        return cm


if __name__ == '__main__':
    # TO RUN WITH ALL DATA, REPLACE THE DATASET FOLDER IN ROOT FOR THE caddy-gestures-complete-v2-release-all-scenarios-fast.ai folder in zip file
    # AND RENAME IT TO dataset.

    classifier = CaddyClassifier(image_shape=(200, 200, 3), re_train=False)
    classifier.evaluate()
