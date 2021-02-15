import os
import csv
import math
import torch
import torch.nn.functional as F
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
# from utils import list_pictures
from torchvision import transforms
from torch.utils.data import dataloader
from PIL import Image

class Data:
    def __init__(self, args):
        self.args = args
        transform_list = [
            transforms.RandomChoice(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(),
                 transforms.RandomRotation(20),
                 ]
            ),
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        transform = transforms.Compose(transform_list)
        self.train_dataset = Dataset(args.train_img, args.train_label, transform)
        self.train_loader = dataloader.DataLoader(self.train_dataset,
                                                  shuffle=True,
                                                  batch_size=args.train_batch_size,
                                                  num_workers=args.nThread
                                                  )


class Dataset(dataset.Dataset):
    def __init__(self, train_img, train_label, transform, flip=False):
        self.root = train_img
        self.transform = transform
        # import ipdb; ipdb.set_trace()
        self.labels = [label[0:-1] for label in csv.reader(open(train_label, 'r'))]
        self.loader = default_loader
        self.flip = flip

    def __getitem__(self, index):
        name, age = self.labels[index]
        age = float (age)
        img = self.loader(os.path.join(self.root, name))
        # import ipdb; ipdb.set_trace()

        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)

        age = int(age)
        if not self.flip:
            if self.transform is not None:
                img = self.transform(img)
            return img, label, age
        else:
            flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.transform is not None:
                img = self.transform(img)
                flip = self.transform(flip)
            return img, flip, label, age

    def __len__(self):
        return len(self.labels)


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)
