import time
import torch
import numpy as np
from tqdm import tqdm

from toolbox import get_model
import json
with open('configs/fpua.json', 'r') as fp:
    cfg = json.load(fp)
net = get_model(cfg)
net.cuda()
#net.convert_to_deploy()

x = torch.zeros((1,3,480,320)).cuda()
t_all = []
net.eval()
for i in tqdm(range(100)):
    t1 = time.time()
    net(x)
    t2 = time.time()
    if i >10:
        t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))
