from functools import partial
import os
import torch
import torch.nn.functional as F
import data
import loss
import utils
import numpy as np
from option import args
from model import ThinAge, TinyAge, resnet18
from test import test
from tqdm import tqdm


models = {'ThinAge': ThinAge, 'TinyAge': TinyAge, 'resnet18':resnet18}


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
    pretrained_fn = 'pretrained/{}.pt'.format(args.model_name)
    if os.path.isfile(pretrained_fn):
        model = torch.load(pretrained_fn)
        print('load pretrained')
    else:
        model = get_model()

    os.makedirs(args.ckpt_dir, exist_ok=1)
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
            # import ipdb; ipdb.set_trace()
            mask = F.one_hot(inputs['mask'].long(), 11).float().permute(0, 3, 1, 2).to(device)
            # img = torch.cat([img, mask], dim=1)
            img = img.to(device)
            label = label.to(device)
            age = age.to(device)
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
        torch.save(model, './{}/{}.pt'.format(args.ckpt_dir, args.model_name))
        torch.save(model.state_dict(),
                   './{}/{}_dict.pt'.format(args.ckpt_dir, args.model_name))
        if (i+1) % 2 == 0:
            print('Test: Epoch=[{}]'.format(i))
            cur_mae = test(model)
            if cur_mae < best_mae:
                best_mae = cur_mae
                print(f'Saving best model with MAE {cur_mae}... ')
                torch.save(
                    model, './{}/best_{}_MAE={}.pt'.format(args.ckpt_dir, args.model_name, cur_mae))
                torch.save(model.state_dict(),
                           './{}/best_{}_dict_MAE={}.pt'.format(args.ckpt_dir, args.model_name, cur_mae))


if __name__ == '__main__':
    main()
