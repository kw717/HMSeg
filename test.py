import os
import time
from tqdm import tqdm
from PIL import Image
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from toolbox import get_dataset
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import class_to_RGB, load_ckpt, save_ckpt
from toolbox.datasets.mdd import MDD



def evaluate(logdir,cfg_path, save_predict=False, options=['train', 'test'], prefix='',pth=''):
    # 加载配置文件cfg
    cfg = None
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda')
    trainset, valset, testset = get_dataset(cfg)

    loaders = []
    for opt in options:

        loaders.append((opt, DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])))

        cmap = testset.cmap

    model = get_model(cfg)
    model.to(device)

    dict = torch.load(pth, map_location='cuda:0')
    dict2 = {key: value for key, value in dict.items() if ('aux' not in key)}
    model.load_state_dict(dict2, strict=False)

    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()
    save_path = os.path.join(logdir, 'predicts')

    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    for name, test_loader in loaders:
        running_metrics_val.reset()
        print('#'*50 + '    ' + name+prefix + '    ' + '#'*50)
        with torch.no_grad():
            model.eval()
            for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):

                time_start = time.time()

                if cfg['output'] == 'm':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)[0]
                else:
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)

                predict = predict.max(1)[1].cpu().numpy()
                label = label.cpu().numpy()

                running_metrics_val.update(label, predict)

                time_meter.update(time.time() - time_start, n=image.size(0))

                if save_predict:
                    predict = predict.squeeze(0)
                    label = label.squeeze(0)
                    label = class_to_RGB(label, N=len(cmap), cmap=cmap)
                    predict = class_to_RGB(predict, N=len(cmap), cmap=cmap)
                    predict = Image.fromarray(predict)
                    label = Image.fromarray(label)
                    predict.save(os.path.join(save_path, sample['label_path'][0]))

        metrics = running_metrics_val.get_scores()
        print('overall metrics .....')
        for k, v in metrics[0].items():
            print(k, f'{v:.4f}')

        print('iou for each class .....')
        for k, v in metrics[1].items():
            print(k, f'{v:.4f}')
        print('acc for each class .....')
        for k, v in metrics[2].items():
            print(k, f'{v:.4f}')



if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", default="./test/", type=str,
                        help="run logdir")
    parser.add_argument("--config", type=str, default="configs/HMSeg.json", help="Configuration file to use")
    parser.add_argument("--pth", default="HMSeg_mdd.pth", type=str,
                        help="checkpoint")
    parser.add_argument("-s", type=bool, default=False,
                        help="save predict or not")

    args = parser.parse_args()

    evaluate(args.logdir,args.config, options=['test'], prefix='',save_predict=args.s,pth=args.pth)
