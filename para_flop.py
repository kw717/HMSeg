from ptflops import get_model_complexity_info
from toolbox import get_model
from mmcv.cnn import get_model_complexity_info
import torch
import json
with open('configs/fpua.json', 'r') as fp:
    cfg = json.load(fp)
model = get_model(cfg)



f,p = get_model_complexity_info(model,(3,480,320))
print(f+'\n'+p)

total = sum([param.nelement() for param in model.parameters()])
print("Number of parameters: %.2fM" % (total/1e6))