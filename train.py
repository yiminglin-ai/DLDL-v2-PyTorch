import os
import torch
import data
import loss
import utils
import numpy as np
from option import args
from model import ThinAge, TinyAge, get_resnet18
from test import test
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_value_

models = {'ThinAge': ThinAge, 'TinyAge': TinyAge, 'resnet18':get_resnet18}


def get_model(pretrained=False):
    model = args.model_name
    assert model in models
    if pretrained:
        path = os.path.join('./pretrained/{}.pt'.format(model))
        assert os.path.exists(path)
        return torch.load(path)
    model = models[model]()

    return model


def main():
    model = get_model()
    device = torch.device('cuda')
    model = model.to(device)
    print(model)
    loader = data.Data(args).train_loader
    rank = torch.Tensor([i for i in range(101)]).cuda()
    best_mae = np.inf
    for i in range(args.epochs):
        lr = 0.001 if i < 30 else 0.0001
        optimizer = utils.make_optimizer(args, model, lr)
        model.train()
        print('Learning rate:{}'.format(lr))
        # start_time = time.time()
        for j, inputs in enumerate(tqdm(loader)):
            img, label, age = inputs['image'], inputs['label'], inputs['age']
            img = img.to(device)
            label = label.to(device)
            age = age.to(device)
            optimizer.zero_grad()
            outputs = model(img)
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
        torch.save(model, './checkpoint/{}.pt'.format(args.model_name))
        torch.save(model.state_dict(),
                   './checkpoint/{}_dict.pt'.format(args.model_name))
        if (i+1) % 2 == 0:
            print('Test: Epoch=[{}]'.format(i))
            cur_mae = test(model)
            if cur_mae < best_mae:
                best_mae = cur_mae
                print(f'Saving best model with MAE {cur_mae}... ')
                torch.save(
                    model, './checkpoint/best_{}_MAE={}.pt'.format(args.model_name, cur_mae))
                torch.save(model.state_dict(),
                           './checkpoint/best_{}_dict_MAE={}.pt'.format(args.model_name, cur_mae))


if __name__ == '__main__':
    main()
