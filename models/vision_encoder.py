import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOVisionEncoder(nn.Module):
    def __init__(self, model_weight='yolov8s.pt', freeze_backbone=True):
        super().__init__()
        
        try:
            # 1. 加载临时包装器
            yolo_wrapper = YOLO(model_weight, task='detect') 
            self.model = yolo_wrapper.model
            del yolo_wrapper
            
            # 2. 修复 train 方法
            if hasattr(self.model, 'train') and not callable(self.model.train):
                try: del self.model.train
                except AttributeError: pass
            if not callable(getattr(self.model, 'train', None)):
                 self.model.train = list(self.model.modules())[0].train

        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise e
            
        self.target_layers = [4, 6, 9] 
        self.features = {}
        
        # 初始注册
        self._force_reregister_hooks()

        # 试运行 (用于获取通道数和检查 Hook 状态)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            self.model(dummy)
            self.out_channels = [self.features[idx].shape[1] for idx in self.target_layers]
            
            # 清理
            self.features = {} 
        
        # print(f"✅ Backbone Channels: {self.out_channels}")

        if freeze_backbone:
            self._freeze_parameters()

    def _get_features_hook(self, layer_id):
        def hook(module, input, output):
            # 这个 self 是闭包绑定的，谁调用 _get_features_hook 就绑定谁
            self.features[layer_id] = output
        return hook

    def _force_reregister_hooks(self):
        """
        强制清理旧钩子并注册新钩子
        """
        for layer_idx in self.target_layers:
            layer = self.model.model[layer_idx]
            # 1. 暴力清空现有的钩子 (解决 EMA 复制导致的僵尸钩子问题)
            if hasattr(layer, '_forward_hooks'):
                layer._forward_hooks.clear()
            
            # 2. 注册绑定当前实例的新钩子
            layer.register_forward_hook(self._get_features_hook(layer_idx))

    def _freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 1. 清空特征容器
        self.features = {}
        
        # 2. 前向传播
        _ = self.model(x)
        
        # 3. 检查 Hook 是否生效 (EMA 自愈机制)
        if len(self.features) < len(self.target_layers):
            self._force_reregister_hooks()
            
            # 再跑一次前向传播
            self.features = {}
            _ = self.model(x)
        
        # 4. 提取特征
        try:
            ordered_features = [self.features[idx] for idx in self.target_layers]
        except KeyError as e:
            print(f"❌ Critical Error: Features still missing. Captured keys: {self.features.keys()}")
            raise e
            
        return ordered_features