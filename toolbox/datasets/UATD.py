import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms

from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, RandomRotation


class UATD(data.Dataset):

    def __init__(self, cfg, mode='trainval', do_aug=False):

        assert mode in ['trainval','val', 'test','train'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomRotation(45),
            RandomCrop(crop_size, pad_if_needed=True)

        ])

        self.val_resize = Resize(crop_size)

        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [6.3040, 4.3505, 35.0686, 3.4997, 14.0079, 8.0937, 32.6272, 28.6828, 14.8280, 38.3528, 37.4353,
                 18.7975])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.2778, 0.1770, 4.7280, 0.1358, 0.7816, 0.3785, 3.7939, 2.5866, 0.8480, 6.5770, 5.8139, 1.2184])
        else:
            raise (f"{cfg['class_weight']} not support.")

        # with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
        #     self.infos = f.readlines()
        if mode == 'train':
            self.infos = os.listdir(os.path.join(self.root,'train','label'))
        if mode == 'test':
            infos1 = os.listdir(os.path.join(self.root,'test1','label'))
            infos2 = os.listdir(os.path.join(self.root,'test2','label'))
            self.infos = []
            for i in infos1:
                self.infos.append(os.path.join('test1','label',i))
            for i in infos2:
                self.infos.append(os.path.join('test2','label',i))


    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()
        if self.mode == 'train':
            imageroot = os.path.join(self.root, 'train','image',  image_path)
            labelroot = os.path.join(self.root, 'train','label',  image_path)
        if self.mode == 'test':
            if os.path.split(image_path)[0] == 'test1/label':
                imageroot = os.path.join(self.root, 'test1', 'image', os.path.split(image_path)[1])
                f = 1
            else:
                imageroot = os.path.join(self.root, 'test2', 'image', os.path.split(image_path)[1])
                f = 2
            labelroot = os.path.join(self.root,  image_path)


        image = Image.open(imageroot)  # RGB 0~255
        label = Image.open(labelroot)  # 1 channel 0~11
        # bound = Image.open(os.path.join(self.root, 'bound', self.mode, image_path))

        # make image 3 channels
        # image = np.asarray(image)
        # image = np.expand_dims(image,axis=0)
        # image = np.concatenate((image,image,image),axis=0)
        # image = Image.fromarray(image)
        image = image.convert('RGB')

        sample = {
            'image': image,
            'label': label,
        }

        if self.mode in ['train'] and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)
        else:
            sample = self.val_resize(sample)
        # if self.mode in ['train', 'trainval'] and self.do_aug:  # 只对训练集增强
        #     sample = self.aug(sample)
        sample['image'] = self.im_to_tensor(sample['image'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        # sample['bound'] = torch.from_numpy(np.asarray(sample['bound'], dtype=np.int64)).long()
        if self.mode == 'train':
              sample['label_path'] = image_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        else:
            sample['label_path'] = str(f) + image_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [
            (0, 0, 0),  # unlabeled

            (128, 128, 128),  # sky
            (128, 0, 0),      # building
            (192, 192, 128),  # pole
            (128, 64, 128),   # road
            (0, 0, 192),      # pavement sidewalk
            (128, 128, 0),    # tree
            (192, 128, 128),  # sign_symbol
            (64, 64, 128),    # fence
            (64, 0, 128),     # car
            (64, 64, 0),      # pedestrian


        ]


if __name__ == '__main__':
    import json

    path = '/home/wangyike/sonar_seg/configs/UATD.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)
    #cfg['root'] = '/home/dtrimina/Desktop/lxy/database/camvid'


    dataset = UATD(cfg, mode='test', do_aug=False)
    print(dataset.__len__())
    from toolbox.utils import class_to_RGB
    import matplotlib.pyplot as plt

    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample['image']
        label = sample['label']

        image = image.numpy()
        image = image.transpose((1, 2, 0))
        image *= np.asarray([0.229, 0.224, 0.225])
        image += np.asarray([0.485, 0.456, 0.406])

        label = label.numpy()
        print(label.max())
        label = class_to_RGB(label, N=len(dataset.cmap), cmap=dataset.cmap)

        #plt.subplot('1,2,1')
        plt.imshow(image)
        #plt.subplot('1,2,2')
        plt.imshow(label)

        plt.show()

        if i == 10:
            break


    # dataset = Camvid(cfg, mode='trainval', do_aug=False)
    # from toolbox.utils import ClassWeight
    #
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['ims_per_gpu'], shuffle=True,
    #                                            num_workers=cfg['num_workers'], pin_memory=True)
    # classweight = ClassWeight('median_freq_balancing')  # enet, median_freq_balancing
    # class_weight = classweight.get_weight(train_loader, cfg['n_classes'])
    # class_weight = torch.from_numpy(class_weight).float()
    # # class_weight[cfg['id_unlabel']] = 0
    #
    # print(class_weight)
    #
    # # # median_freq_balancing
    # # tensor([0.2778, 0.1770, 4.7280, 0.1358, 0.7816, 0.3785, 3.7939, 2.5866, 0.8480,
    # #         6.5770, 5.8139, 1.2184])
    #
    # # # enet
    # # tensor([6.3040, 4.3505, 35.0686, 3.4997, 14.0079, 8.0937, 32.6272, 28.6828,
    # #         14.8280, 38.3528, 37.4353, 18.7975])
