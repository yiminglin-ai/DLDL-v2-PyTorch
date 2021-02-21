from functools import partial
import os
import torch
import torch.nn.functional as F

import loss
import utils
import numpy as np
from option import args
from model import ThinAge, TinyAge, resnet18
from test import test
from tqdm import tqdm
import data
import imdb
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Subset
from torch.utils.data import dataloader
from torch import nn
from classy_vision.models.classy_model import ClassyModel
from classy_vision.generic.util import load_checkpoint
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import torch
import cv2
import pretrainedmodels.models.torchvision_models as models
import ibug.roi_tanh_warping.reference_impl as ref


class Rtnet_FCN_Resnet50(nn.Module):
    def __init__(self, num_classes: int, aux_loss: bool = False, ckpt=None):
        super().__init__()
        self.model = fcn_resnet50(num_classes=num_classes, aux_loss=aux_loss)
        if ckpt:
            classy_state = load_checkpoint(ckpt)
            self.load_state_dict(classy_state['model']['trunk'], False)

    def forward(self, x):
        x = self.model(x)
        return x


def restore_warp(h, w, logits: torch.Tensor, bbox):
    logits = logits.cpu().sigmoid().numpy().transpose(1, 2, 0)
    logits[..., 0] = 1 - logits[..., 0]  # background class
    logits = ref.roi_tanh_polar_restore(
        logits, bbox, (h, w), keep_aspect_ratio=True
    )
    logits[..., 0] = 1 - logits[..., 0]
    predict = np.argmax(logits, -1)
    return logits, predict


ckpt = '/home/yiminglin/ibug/facial-age-estimation/experiments/008_parsing_age/checkpoint.torch'

face_parser = Rtnet_FCN_Resnet50(11, ckpt=ckpt)
face_parser = face_parser.cuda().eval()


models = {'ThinAge': ThinAge, 'TinyAge': TinyAge, 'resnet18': resnet18}


def get_model(pretrained=False):
    model = args.model_name
    assert model in models
    if pretrained:
        path = os.path.join('./pretrained/{}.pt'.format(model))
        assert os.path.exists(path)
        return torch.load(path)
    model = models[model]()

    return model


transform_list = [
    #     A.Resize(height=args.height, width=args.width),
    data.RoITanhPolarWarp(),
    A.Normalize(0.5, 0.5),
    ToTensorV2()
]
# transform = transforms.Compose(transform_list)
transform = A.Compose(transform_list, bbox_params=A.BboxParams(
    format='pascal_voc', label_fields=['category_ids']))


def main():
    model = get_model()
    device = torch.device('cuda')
    model = model.to(device)
    print(model)
    target_size = (512, 512)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30,
                   b_shift_limit=30, p=0.3),
        A.ToGray(),
        data.RoITanhPolarWarp(target_size=target_size, aug_p=0.5),
        # RoITanhWarp(target_size=target_size),
        A.Normalize(),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['category_ids'], ),
    )

    val_transform = A.Compose([
        data.RoITanhPolarWarp(target_size=target_size),
        # RoITanhWarp(target_size=target_size),
        A.Normalize(),
        ToTensorV2()

    ],
        bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['category_ids'],
    ))
    train_idx = np.load('train_idx.npy').astype(int)
    test_idx = np.load('test_idx.npy').astype(int)
    train_set = imdb.AgeLMDB(args.imdb_lmdb, transforms=train_transform)
    test_set = imdb.AgeLMDB(args.imdb_lmdb, transforms=val_transform)

    train_dataset = Subset(train_set, train_idx)
    valid_dataset = Subset(test_set, test_idx)

    loader = dataloader.DataLoader(train_dataset,
                                   shuffle=True,
                                   batch_size=args.train_batch_size,
                                   num_workers=args.nThread)

    val_loader = dataloader.DataLoader(valid_dataset,
                                   shuffle=False,
                                   batch_size=args.train_batch_size,
                                   num_workers=args.nThread)
    rank = torch.Tensor([i for i in range(101)]).cuda()
    best_mae = np.inf
    for i in range(args.epochs):
        lr = 0.001 if i < args.epochs//2 else 0.0001
        optimizer = utils.make_optimizer(args, model, lr)
        face_parser.eval()
        model.train()
        print('Learning rate:{}'.format(lr))
        # start_time = time.time()
        for j, inputs in enumerate(tqdm(loader)):
            img, label, age = inputs['image'], inputs['label'], inputs['age']
            # import ipdb; ipdb.set_trace()
            img = img.to(device)
            with torch.no_grad():
                mask = face_parser(img)['out'].argmax(1)
                mask = F.one_hot(mask.long(), 11).float().permute(
                0, 3, 1, 2).to(device)
                mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')
                img = F.interpolate(img, scale_factor=0.5, mode='bilinear')


            # img = torch.cat([img, mask], dim=1)
            label = label.to(device).float()
            age = age.to(device).float()
            optimizer.zero_grad()
            outputs = model(img, mask)
            ages = torch.sum(outputs*rank, dim=1)
            loss1 = loss.kl_loss(outputs, label)
            loss2 = loss.L1_loss(ages, age)
            total_loss = loss1 + loss2
            total_loss.backward()
            optimizer.step()
            # current_time = time.time()
            if j % 10 == 0:
                tqdm.write('[Epoch:{}] \t[batch:{}]\t[loss={:.4f}]'.format(
                    i, j, total_loss.item()))
            # start_time = time.time()
        torch.save(model, './pretrained/{}.pt'.format(args.model_name))
        torch.save(model.state_dict(),
                   './pretrained/{}_dict.pt'.format(args.model_name))
        if (i+1) % 2 == 0:
            print('Test: Epoch=[{}]'.format(i))
            cur_mae = evaluate_model(model, val_loader, device, rank)
            if cur_mae < best_mae:
                best_mae = cur_mae
                print(f'Saving best model with MAE {cur_mae}... ')
                torch.save(
                    model, './pretrained/best_{}_MAE={}.pt'.format(args.model_name, cur_mae))
                torch.save(model.state_dict(),
                           './pretrained/best_{}_dict_MAE={}.pt'.format(args.model_name, cur_mae))
@torch.no_grad()
def evaluate_model(model, dl, device, rank):
    model.eval()
    mae = 0.0
    total = 0.0
    for j, inputs in enumerate(tqdm(dl)):
        img, label, age = inputs['image'], inputs['label'], inputs['age']
        # import ipdb; ipdb.set_trace()
        img = img.to(device)
        with torch.no_grad():
            mask = face_parser(img)['out'].argmax(1)
            mask = F.one_hot(mask.long(), 11).float().permute(
                0, 3, 1, 2).to(device)
            mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')
            img = F.interpolate(img, scale_factor=0.5, mode='bilinear')
        # img = torch.cat([img, mask], dim=1)
        label = label.to(device).float()
        age = age.to(device).float()

        outputs = model(img, mask)
        ages = torch.sum(outputs*rank, dim=1)
        mae += (ages-age).abs().sum().item()
        total += len(img)
    return mae/total
if __name__ == '__main__':
    main()
