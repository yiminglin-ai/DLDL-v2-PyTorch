import torch
import csv
import numpy as np
import os
from option import args
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import data
from torch.utils.data import dataloader
group = {0: "0-19", 1: "20-29", 2: "30-39",
         3: "40-49", 4: "50-59", 5: "60-69", 6: "70-"}


def get_group(age):
    if 0 <= age <= 19:
        return 0
    elif 20 <= age <= 29:
        return 1
    elif 30 <= age <= 39:
        return 2
    elif 40 <= age <= 49:
        return 3
    elif 50 <= age <= 59:
        return 4
    elif 60 <= age <= 69:
        return 5
    elif 70 <= age:
        return 6
    else:
        raise ValueError


BATCH_SIZE = 64
transform_list = [
    transforms.Resize((args.height, args.width), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
transform = transforms.Compose(transform_list)


def preprocess(img):
    img = Image.open(img).convert('RGB')
    imgs = [img, img.transpose(Image.FLIP_LEFT_RIGHT)]
    imgs = [transform(i) for i in imgs]
    imgs = [torch.unsqueeze(i, dim=0) for i in imgs]
    return imgs


def test():
    num_group = len(group)
    group_count = torch.zeros(num_group)

    model = torch.load('./pretrained/{}.pt'.format(args.model_name))
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    rank = torch.Tensor([i for i in range(101)]).cuda()
    error = 0
    count = 0
    correct_count = torch.zeros(num_group)
    correct_group = torch.zeros(num_group)

    test_dataset = data.Dataset(
        args.val_img, args.val_label, transform, flip=True)
    test_loader = dataloader.DataLoader(test_dataset,
                                        shuffle=False,
                                        batch_size=args.train_batch_size,
                                        num_workers=args.nThread)

    with torch.no_grad():
        for i, inputs in enumerate(tqdm(test_loader)):
            img, flip, label, age = inputs
            for p in age:
                group_count[get_group(p.item())] += 1
            img = img.to(device)
            flip = flip.to(device)
            label = label.to(device)
            age = age.to(device)
            outputs = model(img)
            outputs_flip = model(flip)
            # predict_age = 0
            predict_age = torch.sum(outputs*rank, dim=1) + \
                torch.sum(outputs_flip*rank, dim=1)
            predict_age = predict_age / 2.
            # predict_age = predict_age.sum()
            # if count % 10==0:
            #     tqdm.write('label:{} \tage:{:.2f}'.format(age, predict_age))
            error += (predict_age-age).abs().sum().item()
            count += len(img)
            for ind, a in enumerate(predict_age):
                if abs(age[ind].item() - a) < 1:
                    correct_count[get_group(age[ind].item())] += 1
                    correct_group[get_group(age[ind].item())] += 1
                elif get_group(age[ind].item()) == get_group(a):
                    correct_group[get_group(age[ind].item())] += 1

    for ind, p in enumerate(group_count):
        if p == 0:
            group_count[ind] = 1
    MAE = error / count
    print('MAE:{:.4f}'.format(MAE))
    print("\nCorrect group rate:")
    print(correct_group/group_count)
    print("Correct age rate:")
    print(correct_count/group_count)
    rate = (correct_group, correct_count)

    return MAE


if __name__ == '__main__':
    test()
