import sys
sys.dont_write_bytecode = True

import seaborn as sns
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
import os
from os.path import exists
import matplotlib.pyplot as plt
from data import *
from torchsummary import summary
import copy
import time

def extract_zip_folder(zip_path: str, folder_name) -> None:
    """Extracts a zip folder to a given path.

    Args:
        zip_folder_path (str): name of the zip folder.
    """
    zip = zipfile.ZipFile(file=zip_path, mode='r')
    zip.extractall(path=os.path.join(os.path.dirname(zip_path), folder_name))
    zip.close()

class CaddyClassifier:
    def __init__(self, image_shape: tuple[int, int, int], optimizer = 'adam', batch_size: int = 3, re_train: bool = False) -> None:
        self._image_shape = image_shape
        self._batch_size = batch_size
        self._model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model')
        self._training_dataset, self._validation_dataset, self._test_dataset = self._load_datasets()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.max_memory_allocated(device=self._device)
        self._optimizer = optimizer
        self._model: models.ResNet = self._load(re_train=re_train)

    def _load_datasets(self) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
        training_dataset = self._create_dataset('train')
        validation_dataset = self._create_dataset('val')
        test_dataset = self._create_dataset('test')

        return training_dataset, validation_dataset, test_dataset

    def _create_dataset(self, set:str) -> torch.utils.data.DataLoader:
        compose = transforms.Compose([
			transforms.Resize(self._image_shape),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        img_dataset = datasets.ImageFolder(os.path.join(IMG_DIR, set), compose)
        self._classes = img_dataset.classes
        dataset = torch.utils.data.DataLoader(img_dataset, batch_size=self._batch_size, shuffle=True, num_workers=1)

        return dataset

    def _load(self, re_train: bool) -> models.ResNet:
        if exists(os.path.join(self._model_path, 'resNetcaddyClassifier.pickle')):
            if re_train:
                classifier = self._train()
            else:
                classifier = pickle.load(open(os.path.join(self._model_path, 'resNetcaddyClassifier.pickle'), "rb"))
        else:
            classifier = self._train()

        return classifier

    def _train(self) -> models.ResNet:
        model = self._compile_model()

        criterion = nn.CrossEntropyLoss()

        if self._optimizer == 'sgd':
            optimizer_ft = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
        elif self._optimizer == 'adam':
            optimizer_ft = optim.Adam(model.parameters(), lr=0.0005)

        scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.01)

        since = time.time()

        print('copying the model')
        best_model_wts = copy.deepcopy(model.state_dict())
        print('done with deep copy')
        best_acc = 0.0

        num_epochs = 50
        dataloaders = {'train':self._training_dataset, 'val':self._validation_dataset}
        dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
        print('entering training loop')
        for epoch in range(num_epochs):
           print('Epoch {}/{}'.format(epoch, num_epochs - 1))
           print('-' * 10)

           for phase in ['train', 'val']:
               if phase == 'train':
                   scheduler.step()
                   model.train()
               else:
                   model.eval()

               running_loss = 0.0
               running_corrects = 0.0

               for inputs, labels in dataloaders[phase]:
                   inputs = inputs.to(self._device)
                   labels = labels.to(self._device)

                   optimizer_ft.zero_grad()

                   with torch.set_grad_enabled(phase == 'train'):
                       outputs = model(inputs)
                       logits, preds = torch.max(outputs, 1)
                       loss = criterion(outputs, labels)

                       if phase == 'train':
                           loss.backward()
                           optimizer_ft.step()

                   running_loss += loss.item() * inputs.size(0)
                   running_corrects += torch.sum(preds == labels.data)

               epoch_loss = running_loss / dataset_sizes[phase]
               epoch_acc = running_corrects.double() / dataset_sizes[phase]

               print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

               if phase == 'val' and epoch_acc > best_acc:
                   best_acc = epoch_acc
                   best_model_wts = copy.deepcopy(model.state_dict())

           latest_model = copy.deepcopy(model.state_dict())
           model_path = os.path.join(os.getcwd(), "model", "resNetcaddyClassifier.pickle")
           pickle.dump(latest_model, open(model_path, "wb"))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
        model_path = os.path.join(os.getcwd(), "model", "resNetcaddyClassifier.pickle")
        pickle.dump(model, open(model_path, "wb"))

        return model

    def _compile_model(self) -> models.ResNet:
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 17)
        model = model.to(self._device)
        
        X, Y = self._image_shape
        summary(model, input_size=(3, X, Y))

        return model

    def evaluate(self) -> np.ndarray:
        nb_classes = 17
        confusion_matrix = torch.zeros(nb_classes, nb_classes)

        with torch.no_grad():
            for i, (inputs, classes) in enumerate(self._test_dataset):
                inputs = inputs.to(self._device)
                classes = classes.to(self._device)
                outputs = self._model(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        confusion_matrix.to(torch.int32)
        confusion_matrix = confusion_matrix.numpy()
        
        for i in range(len(confusion_matrix)):
            confusion_matrix[i] = confusion_matrix[i]/sum(confusion_matrix[i])

        precisions = []
        recalls = []
        for i in range(len(confusion_matrix)):
            precisions.append(confusion_matrix[i][i]/sum(confusion_matrix[:,i]))
            recalls.append(confusion_matrix[i][i]/sum(confusion_matrix[i]))

        test_classes = ['backward', 'boat', 'carry', 'delimiter', 'down', 'end', 'five', 'four', 'here', 'mosaic', 'none', 'one', 'photo', 'start', 'three', 'two', 'up']

        plt.figure(figsize=(16, 10))
        plt.barh(test_classes, precisions)
        plt.title('Precision')
        plt.ylabel('Precision')
        plt.xlabel('Classes')
        for i, v in enumerate(precisions):
            plt.text(v, i, str(round(v, 2)), color='black', va='center')
        plt.show()

        plt.figure(figsize=(16, 10))
        plt.barh(test_classes, recalls)
        plt.title('Recall')
        plt.ylabel('Recall')
        plt.xlabel('Classes')
        for i, v in enumerate(recalls):
            plt.text(v, i, str(round(v, 2)), color='black', va='center')
        plt.show()

        plt.figure(figsize=(16, 10))
        sns.heatmap(confusion_matrix, annot=True, cmap='Blues', yticklabels=test_classes, xticklabels=test_classes)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        return confusion_matrix

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(cur_dir, 'images')

    if not exists(os.path.join(images_dir, 'train')) or not exists(os.path.join(images_dir, 'test')):
        if (not exists(os.path.join(images_dir, 'test.zip')) or not exists(os.path.join(images_dir, 'train.zip'))):
            raise Exception('Please download the dataset #PUT DRIVE LINK HERE')

        extract_zip_folder(os.path.join(images_dir, 'test.zip'), 'test')
        extract_zip_folder(os.path.join(images_dir, 'train.zip'), 'train')

    if not exists(os.path.join(images_dir, 'val')):
        data_augmentation(images_dir, NUM_CLASSES=NUM_CLASSES, CLASSES=CLASSES)
        split_train_val()
        preprocess_images(grayscale)

    get_image_count(IMG_DIR=IMG_DIR, CLASSES=CLASSES, NUM_CLASSES=NUM_CLASSES)
    classifier = CaddyClassifier(image_shape=(500, 500), re_train=False)
    classifier.evaluate()
