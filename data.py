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
import cv2
import numpy as np
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
import ibug.roi_tanh_warping.reference_impl as ref

from albumentations.augmentations import functional as F
from albumentations.core.transforms_interface import to_tuple
import random

def bbox_shift_scale(bbox, scale, dx, dy, **kwargs):  # skipcq: PYL-W0613
    x_min, y_min, x_max, y_max = bbox[:4]  # pascal_voc
    center = (x_min+x_max) / 2., (y_min + y_max) / 2.
    matrix = cv2.getRotationMatrix2D(center, 0, scale)
    matrix[0, 2] += dx * (x_max-x_min)
    matrix[1, 2] += dy * (y_max-y_min)
    x = np.array([x_min, x_max, x_max, x_min])
    y = np.array([y_min, y_min, y_max, y_max])
    ones = np.ones(shape=(len(x)))
    points_ones = np.vstack([x, y, ones]).transpose()
    tr_points = matrix.dot(points_ones.T).T
    x_min, x_max = min(tr_points[:, 0]), max(tr_points[:, 0])
    y_min, y_max = min(tr_points[:, 1]), max(tr_points[:, 1])
    return x_min, y_min, x_max, y_max


class RoITanhPolarWarp(A.DualTransform):
    def __init__(self,
                 target_size=(512, 512),
                 interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_CONSTANT,
                 border_value=0,
                 keep_aspect_ratio=True,
                 always_apply=False, p=1,
                 aug_p=0,
                 shift_limit=0.1,
                 scale_limit=0.1,
                 rotate_limit=30,
                 shift_limit_x=None,
                 shift_limit_y=None
                 ):
        super(RoITanhPolarWarp, self).__init__(always_apply, p)
        if isinstance(target_size, str):
            target_size = eval(target_size)
        self.target_size = target_size
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value
        self.keep_aspect_ratio = keep_aspect_ratio
        self.shift_limit_x = to_tuple(
            shift_limit_x if shift_limit_x is not None else shift_limit)
        self.shift_limit_y = to_tuple(
            shift_limit_y if shift_limit_y is not None else shift_limit)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.rotate_limit = to_tuple(rotate_limit)
        self.aug_p = aug_p

    def apply(self, image, bbox_, angle=0, scale=0, dx=0, dy=0, aug_p=0, **params):
        if self.aug_p > aug_p:
            bbox_ = bbox_shift_scale(bbox_, scale, dx, dy, **params)
        else:
            angle = 0
        angle = angle / 180. * np.pi
        image = ref.roi_tanh_polar_warp(image, bbox_, self.target_size,
                                        angle,
                                        self.interpolation,
                                        self.border_mode,
                                        self.border_value,
                                        keep_aspect_ratio=self.keep_aspect_ratio)
        return image

    # def apply_to_mask(self, img, bbox, **params):
    def apply_to_mask(self, image, bbox_, angle=0, scale=0, dx=0, dy=0, aug_p=0, cols=0, rows=0,**params):
        if self.aug_p > aug_p:
            bbox_ = bbox_shift_scale(bbox_, scale, dx, dy, **params)
        else:
            angle = 0
        angle = angle / 180. * np.pi
        image = ref.roi_tanh_polar_warp(image, bbox_, self.target_size,
                                        angle,
                                        cv2.INTER_NEAREST,
                                        self.border_mode,
                                        self.border_value,
                                        keep_aspect_ratio=self.keep_aspect_ratio)
        return image

    def get_params_dependent_on_targets(self, params):
        bbox_ = params["bboxes"][0]
        h,w = params['image'].shape[:2]
        bbox_ = A.convert_bbox_from_albumentations(bbox_, 'pascal_voc', h, w)
        return {"bbox_": bbox_}
    
    def apply_to_bbox(self, bbox, **params):
        return bbox

    @property
    def targets_as_params(self):
        return ["bboxes", 'image']

    def get_params(self):
        return {
            "angle": random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
            "scale": random.uniform(self.scale_limit[0], self.scale_limit[1]),
            "dx": random.uniform(self.shift_limit_x[0], self.shift_limit_x[1]),
            "dy": random.uniform(self.shift_limit_y[0], self.shift_limit_y[1]),
            "aug_p": np.random.uniform()
        }

    def get_transform_init_args_names(self):
        return {
            "target_size": self.target_size,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
            "border_value": self.border_value,
            "keep_aspect_ratio": self.keep_aspect_ratio,
            "shift_limit_x": self.shift_limit_x,
            "shift_limit_y": self.shift_limit_y,
            "scale_limit": to_tuple(self.scale_limit, bias=-1.0),
            "rotate_limit": self.rotate_limit,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
            "aug_p": self.aug_p,
        }



def bbox_hflip(bbox, cols):  # skipcq: PYL-W0613
    """Flip a bounding box horizontally around the y-axis.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    return cols - x_max, y_min, cols - x_min, y_max
                                                  
class Data:
    def __init__(self, args):
        self.args = args
        # transform_list = [
        #     transforms.RandomChoice(
        #         [transforms.RandomHorizontalFlip(),
        #          transforms.RandomGrayscale(),
        #          transforms.RandomRotation(20),
        #          ]
        #     ),
        #     transforms.Resize((args.height, args.width)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ]
        # transform = transforms.Compose(transform_list)
        transform_list = [
            A.OneOf(
                [A.HorizontalFlip(),
                 A.ToGray(),
                 A.Rotate(20)]
            ),
            # A.Resize(height=args.height, width=args.width),
            RoITanhPolarWarp([args.height, args.width], aug_p=0.5),
            A.Normalize(),
            # 
            ToTensorV2()

            # transforms.RandomChoice(
            #     [transforms.RandomHorizontalFlip(),
            #      transforms.RandomGrayscale(),
            #      transforms.RandomRotation(20),
            #      ]
            # ),
            # transforms.Resize((args.height, args.width)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        # transform = transforms.Compose(transform_list)
        transform = A.Compose(transform_list, bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['category_ids']))

        self.train_dataset = Dataset(args.train_img, args.train_label, args.detect_dir, transform)
        self.train_loader = dataloader.DataLoader(self.train_dataset,
                                                  shuffle=True,
                                                  batch_size=args.train_batch_size,
                                                  num_workers=args.nThread)



class Dataset(dataset.Dataset):
    def __init__(self, train_img, train_label, detect_dir, transform, flip=False):
        self.root = train_img
        self.transform = transform
        self.detect_dir = detect_dir
        self.mask_dir = detect_dir.replace('detected', 'parsed')
        self.labels = [label[0:-1] for label in csv.reader(open(train_label, 'r'))]
        self.loader = default_loader
        self.flip = flip

    def __getitem__(self, index):
        name, age = self.labels[index]
        age = float (age)
        img = self.loader(os.path.join(self.root, name))
        w, h = img.width, img.height

        # import ipdb; ipdb.set_trace()
        bbox = np.loadtxt(os.path.join(
            self.detect_dir, name[:-3]+'bbox.txt'), int)
        mask = cv2.imread(os.path.join(
            self.mask_dir, name[:-3]+'png'), cv2.IMREAD_GRAYSCALE)
        x1, y1, x2, y2 = bbox[:4]
        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        y1 = np.clip(y1, 0, h)
        y2 = np.clip(y2, 0, h)
        bbox = list(map(int, [x1, y1, x2, y2]))
        # bbox = [0,0, w-2, h-2]
        
        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)

        age = int(age)

        sample = {
            "image": np.array(img),
            'age':age,
            'w': w,
            'h': h,
            'name': name,
            'label': label,
            "bboxes": [bbox],
            "category_ids":[age],
            "mask": mask,
        }

        if not self.flip:
            if self.transform is not None:
                sample = self.transform(**sample)
            return sample
        else:
            # sample_flip = sample.copy()
            # for roi tanh polar 
            bboxes_ = A.convert_bboxes_to_albumentations(sample['bboxes'], 'pascal_voc', h, w)
            sample_flip = sample.copy()
            sample_flip['bboxes'] = bboxes_
            sample_flip = A.HorizontalFlip(p=1)(**sample_flip)
            sample_flip['bboxes'] = A.convert_bboxes_from_albumentations(sample_flip['bboxes'], 'pascal_voc', h, w)
            # sample_flip['image'] = self.transform(sample['image'])
            # for fake bbox
            # sample_flip['image'] = A.functional.hflip(sample['image'])
            if self.transform is not None:
                sample = self.transform(**sample)
                sample_flip = self.transform(**sample_flip)

            return sample, sample_flip
    def __len__(self):
        return len(self.labels)


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)
