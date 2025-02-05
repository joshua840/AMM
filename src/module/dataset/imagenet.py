import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os


class ImageNet:
    def __init__(self, data_dir):
        self.traindir = os.path.join(data_dir, "train")
        self.valdir = os.path.join(data_dir, "val")
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def get_train_dataset(self):
        train_dataset = datasets.ImageFolder(
            self.traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            ),
        )
        return train_dataset

    def get_val_dataset(self):
        val_dataset = datasets.ImageFolder(
            self.valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            ),
        )
        return val_dataset
