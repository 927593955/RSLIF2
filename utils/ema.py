import math
from copy import deepcopy
import torch
import torch.nn as nn

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from YOLOv5/7 implementation """
    def __init__(self, model, decay=0.9999, updates=0):
        # 创建一个模型的影子副本 (Shadow Copy)
        # eval() 模式，不需要梯度
        self.ema = deepcopy(model).eval()  
        self.updates = updates  # number of EMA updates
        
        # decay 指数衰减率 (e.g. 0.9999)
        # tau 用于动态调整 decay，初期 decay 小，后期大
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000)) 
        
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            
            # state_dict 是参数字典 (weights + bias)
            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Copy attributes from model to ema (e.g., model.names, model.hyp)
        copy_attr(self.ema, model, include, exclude)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        setattr(a, k, v)