from .metrics import averageMeter, runningScore
from .log import get_logger
from .optim import Ranger

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['mdd', 'UATD']

    if cfg['dataset'] == 'mdd':
        from .datasets.mdd import MDD
        return MDD(cfg, mode='train'), MDD(cfg, mode='val'), MDD(cfg, mode='test')
    if cfg['dataset'] == 'UATD':
        from .datasets.UATD import UATD
        return UATD(cfg, mode='train'), UATD(cfg, mode='test'), UATD(cfg, mode='test')


def get_model(cfg):
    if cfg['model_name'] == 'Unet34':
        from segmentation_models_pytorch import Unet
        return Unet(classes=cfg['n_classes'])
    if cfg['model_name'] == 'deeplabv3':
        from segmentation_models_pytorch import DeepLabV3
        return DeepLabV3(classes=cfg['n_classes'])
    if cfg['model_name'] == 'pspnet':
        from segmentation_models_pytorch import PSPNet
        return PSPNet(classes=cfg['n_classes'])
    if cfg['model_name'] == 'linknet':
        from segmentation_models_pytorch import Linknet
        return Linknet(classes=cfg['n_classes'])
    if cfg['model_name'] == 'Unet18':
        from segmentation_models_pytorch import Unet
        return Unet(encoder_name='resnet18', classes=cfg['n_classes'])
    if cfg['model_name'] == 'maanu':
        from .models.MaanuNet.MaanuNet import MaanuNet
        return MaanuNet(3, cfg['n_classes'])
    if cfg['model_name'] == 'fpua':
        from .models.FPUA.FPUA import FPUA
        return FPUA(3, cfg['n_classes'])
    if cfg['model_name'] == 'mitunet':
        from .models.MitUnet.Mit_Unet import MiT_Unet
        return MiT_Unet(cfg['n_classes'])
    if cfg['model_name'] == 'HMSeg':
        from .models.HMNet import HMNet_seg
        return HMNet_seg( cfg['n_classes'])
