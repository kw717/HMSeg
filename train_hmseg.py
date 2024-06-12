import os
import shutil
import json
import time

# from apex import amp

import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam, AdamW,SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from toolbox import get_dataset  # loss
from toolbox.optim.Ranger import Ranger
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import save_ckpt
from toolbox import setup_seed

# from toolbox.losses import LovaszSoftmax # loss
from toolbox.losses import lovasz_softmax

from kornia.filters import box_blur,median_blur,gaussian_blur2d
# loss = lovasz_softmax(out, labels, ignore=255)
import warnings
warnings.filterwarnings('ignore')


setup_seed(33)
class lovasz_softmax_loss(nn.Module):
    def  __init__(self):
        super().__init__()
    def forward(self,out,gt):
        return lovasz_softmax(F.softmax(out, dim=1), gt, ignore=255)
def loss_choose(name):
    if name == 'ce':
        return nn.CrossEntropyLoss()
    if name=='ls':
        return lovasz_softmax_loss()
class eeemodelLoss(nn.Module):

    def __init__(self, class_weight=None, mloss1 = 'ce',mloss2='ls',aloss='ce',ml1=1,ml2=1,al=0.6,dn=0.5 ,ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
            [3,3,2,5,5,3,5,5,1.5,1])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.5121, 10.2388])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4459, 23.7228])).float()
        self.ml1,self.ml2,self.al ,self.dn= ml1,ml2,al,dn
        self.mloss1,self.mloss2,self.aloss = mloss1,mloss2,aloss

        self.class_weight = class_weight
        # self.LovaszSoftmax = lovasz_softmax()
        #self.cross_entropy = nn.CrossEntropyLoss()
        self.loss1 = loss_choose(mloss1)
        self.loss2 = loss_choose(mloss2)
        self.semantic_loss2 = loss_choose(aloss)
        self.L2loss = nn.MSELoss()
        #self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)

    def forward(self, inputs, targets,dnimg):
        semantic_gt = targets
        semantic_out, sal_out,pimg = inputs

        loss1 = self.loss1(semantic_out, semantic_gt)
        loss2 = self.loss2(semantic_out, semantic_gt)
        #loss3 = self.semantic_loss(semantic_out_2, semantic_gt)
        loss3 = self.semantic_loss2(sal_out, semantic_gt)
        loss4 = self.L2loss(pimg,dnimg)


        loss = self.ml1*loss1 + self.ml2*loss2 + self.al * loss3 + self.dn*loss4

        return loss

def run(args):
    torch.cuda.set_device(args.cuda)
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    model = get_model(cfg)
    device = torch.device(f'cuda:{args.cuda}')
    model.to(device)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # device_ids = range(torch.cuda.device_count())
    # device = torch.device('cuda')
    #
    # model = torch.nn.DataParallel(model).cuda()

    trainset, valset, testset = get_dataset(cfg)

    # batchsize 4 numworkers 4
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(valset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                            pin_memory=True)
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                             pin_memory=True)

    params_list = model.parameters()

    if cfg['optim'] == "ranger":
        enc_p = model.encoder.parameters()
        dec_p = model.decoder.parameters()
        aux_p = model.aux.parameters()
        optimizer = Ranger([
            {"params": enc_p, "lr": cfg['lr_start']},
            {"params": dec_p, "lr": cfg['lr_start']*10},
            #{"params": aux_p, "lr": cfg['lr_start'] }
        ], lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])

    elif cfg['optim'] == "adam":
        optimizer = Adam(params_list, lr=cfg['lr_start'], betas=(0.9, 0.999))
    elif cfg['optim'] == "adamw":
        enc_p = model.encoder.parameters()
        dec_p = model.decoder.parameters()

        optimizer = torch.optim.AdamW([
            {"params": enc_p, "lr": 0.00006},
            {"params": dec_p, "lr": 0.0006}
        ], betas=(0.9, 0.999), weight_decay=0.001)
    elif cfg['optim'] == "sgd":
        optimizer = SGD(params_list, lr=cfg['lr_start'], weight_decay=4e-5,momentum=0.9)

        # optimizer = AdamW(params_list,lr=0.0001, weight_decay=0.0001)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    train_criterion = eeemodelLoss(mloss1=cfg["L1"],mloss2=cfg["L2"],aloss=cfg["L3"],ml1 = cfg["loss1"],ml2=cfg["loss2"],al=cfg["loss3"],dn=cfg["dn"]).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    best_test = 0

    # amp.register_float_function(torch, 'sigmoid')
    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            image = sample['image'].to(device)
            label = sample['label'].to(device)

            targets = label
            predict = model(image)
            #dnimg = gaussian_blur2d(image,(3,3),(1.5,1.5))[:,0,:,:]
            dnimg = box_blur(image,(3,3))

            loss = train_criterion(predict, targets,dnimg)
            ####################################################

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        scheduler.step(ep)


        # test
        if ep <= 5 or ep % 3 == 0 or ep >= cfg['epochs'] * 0.5:
            with torch.no_grad():
                model.eval()
                running_metrics_test.reset()
                test_loss_meter.reset()
                for i, sample in enumerate(test_loader):
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)[0]

                    loss = criterion(predict, label)
                    test_loss_meter.update(loss.item())

                    predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                    label = label.cpu().numpy()
                    try:
                        running_metrics_test.update(label, predict)
                    except:
                        pass

            train_loss = train_loss_meter.avg
            test_loss = test_loss_meter.avg

            test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
            test_miou = running_metrics_test.get_scores()[0]["mIou: "]
            test_avg = (test_macc + test_miou) / 2

            logger.info(
                f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] loss={train_loss:.3f}/{test_loss:.3f}, mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}, best={best_test:.3f}')
            if test_miou > best_test:
                best_test = test_miou
                save_ckpt(logdir, model, prefix=str(cfg["model_name"]))

    os.rename(logdir,logdir+str(best_test))


if __name__ == '__main__':
    import argparse
    import torch.backends.cudnn as cudnn

    cudnn.deterministic = True
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="configs/HMSeg.json", help="Configuration file to use") #abexp_wts
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbd'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--cuda", type=int, default=0, help="set cuda device id")

    args = parser.parse_args()

    run(args)


