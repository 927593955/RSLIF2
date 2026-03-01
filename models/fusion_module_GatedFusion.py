import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 保持引用本地模块
from models.yolo_backbone import YOLOv8Backbone, C2f, Conv, Detect 
from models.text_encoder import TriStreamDeBERTa

# ==============================================================================
# 1. 核心改进：空间+通道双重门控融合 (Spatial-Channel Gated Fusion)
# ==============================================================================
class GatedFusion(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        # 通道注意力投影
        self.distill_channel = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.Sigmoid()
        )
        
        # 空间注意力投影：将文本转换为视觉特征的“滤波器”
        self.distill_spatial = nn.Linear(text_dim, visual_dim)
        
        # 融合后的细化卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual_feat, text_feat):
        """
        visual_feat: [B, C, H, W]
        text_feat: [B, D] 全局文本向量
        """
        b, c, h, w = visual_feat.shape
        
        # 1. 通道门控：决定“看什么类别” (Channel-wise)
        # [B, D] -> [B, C, 1, 1]
        c_gate = self.distill_channel(text_feat).view(b, c, 1, 1)
        
        # 2. 空间门控：决定“看哪里” (Spatial-wise) - 解决方位词的关键
        # [B, D] -> [B, C, 1, 1]
        s_query = self.distill_spatial(text_feat).view(b, c, 1, 1)
        # 计算每个像素点与文本的相关性，生成 [B, 1, H, W] 的注意力图
        s_gate = torch.sigmoid(torch.sum(visual_feat * s_query, dim=1, keepdim=True))
        
        # 3. 双重门控引导
        # 注意：此处移除了 + visual_feat 的残差连接，强制模型依赖文本信号
        gated_feat = visual_feat * c_gate * s_gate
        
        return self.fusion_conv(gated_feat)

# ==============================================================================
# 2. 文本特征池化
# ==============================================================================
class TextFeaturePyramid(nn.Module):
    def __init__(self, text_dim=1024):
        super().__init__()

    def forward(self, text_feat, mask=None):
        # text_feat: [B, L, D] -> [B, D]
        if text_feat.dim() == 3:
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                sum_mask = mask_expanded.sum(dim=1) + 1e-6
                global_text = (text_feat * mask_expanded).sum(dim=1) / sum_mask
            else:
                global_text = text_feat.mean(dim=1)
        else:
            global_text = text_feat
        return global_text

# ==============================================================================
# 3. 深度融合 Neck (PAFPN)
# ==============================================================================
class DeepFusionPAFPN(nn.Module):
    def __init__(self, in_channels=[128, 256, 512], text_dim=1024):
        super().__init__()
        c3, c4, c5 = in_channels
        self.text_pooler = TextFeaturePyramid(text_dim)

        # 每一层级都注入空间/通道门控
        self.fuse_p5 = GatedFusion(c5, text_dim) 
        self.fuse_p4 = GatedFusion(c4, text_dim)
        self.fuse_p3 = GatedFusion(c3, text_dim)
        self.fuse_n4 = GatedFusion(c4, text_dim)
        self.fuse_n5 = GatedFusion(c5, text_dim)

        self.reduce_p5 = Conv(c5, c4, 1, 1) 
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p4 = C2f(c4, c4, n=3, shortcut=False) 

        self.reduce_p4 = Conv(c4, c3, 1, 1)
        self.c2f_p3 = C2f(c3, c3, n=3, shortcut=False)

        self.down_p3 = Conv(c3, c3, 3, 2)
        self.c2f_n4 = C2f(c3 + c4, c4, n=3, shortcut=False) 
        
        self.down_p4 = Conv(c4, c4, 3, 2)
        self.c2f_n5 = C2f(c4 + c5, c5, n=3, shortcut=False) 

    def forward(self, raw_p3, raw_p4, raw_p5, combined_text, mask):
        # 1. 获取全局文本向量 [B, 1024]
        global_text_vec = self.text_pooler(combined_text, mask)

        # 2. 自顶向下路径 (语义注入)
        f5 = self.fuse_p5(raw_p5, global_text_vec) 
        f5_up = self.up(self.reduce_p5(f5))

        f4 = self.c2f_p4(f5_up + self.fuse_p4(raw_p4, global_text_vec))
        f4_up = self.up(self.reduce_p4(f4))
        
        f3 = self.c2f_p3(f4_up + self.fuse_p3(raw_p3, global_text_vec))

        # 3. 自底向上路径 (定位强化)
        n3_down = self.down_p3(f3)
        n4 = self.c2f_n4(torch.cat([n3_down, f4], dim=1))
        n4 = self.fuse_n4(n4, global_text_vec) 

        n4_down = self.down_p4(n4)
        n5 = self.c2f_n5(torch.cat([n4_down, f5], dim=1))
        n5 = self.fuse_n5(n5, global_text_vec) 

        return [f3, n4, n5]

# ==============================================================================
# 4. 主模型 (RemoteSensingVLM)
# ==============================================================================
class RemoteSensingVLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = TriStreamDeBERTa(model_path=cfg['deberta_path'])
        self.text_dim = self.text_encoder.hidden_dim
        
        # YOLO Backbone 输出通常为 [128, 256, 512] (针对 width=0.5)
        self.backbone = YOLOv8Backbone(width=0.5, depth=0.33)
        self.neck = DeepFusionPAFPN(in_channels=self.backbone.out_channels, text_dim=self.text_dim)

        nc = cfg.get('nc', 20)
        self.head = Detect(nc=nc, ch=[128, 256, 512])
        self.head.stride = torch.tensor([8.0, 16.0, 32.0])
        
        # 视觉-文本对齐投影层 (对齐最终特征图与语义分支向量)
        self.vis_projector = nn.Conv2d(512, self.text_dim, kernel_size=1)

        # 初始化
        self.apply(self._init_weights)
        self._initialize_biases(self.head)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _initialize_biases(self, m):
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            b[-1].bias.data[:] = -5.0 # 分类分支初始化
            a[-1].bias.data[:] = 1.0  # 回归分支初始化

    def forward(self, imgs, texts):
        # 1. 文本编码 (三分支获取)
        txt_out = self.text_encoder(texts)
        sem, spa, attr, mask = txt_out['sem'], txt_out['spa'], txt_out['attr'], txt_out['mask']
        
        # 融合三分支全局信息作为 Gate 输入
        # 这里建议赋予 spa 分支更高的注意力响应（如果 spa 训练得好）
        combined_text = sem + spa + attr
        
        # 2. 视觉编码 + 深度融合
        raw_feats = self.backbone(imgs) 
        fused_feats = self.neck(raw_feats[0], raw_feats[1], raw_feats[2], combined_text, mask)
        
        # 3. 对齐特征提取
        vis_align_feat = self.vis_projector(fused_feats[-1]) 
        
        # 4. 检测头输出
        head_output = self.head(fused_feats)

        return {
            'feats': head_output, 
            'visual_feats': [vis_align_feat], 
            'sem': sem, 'spa': spa, 'attr': attr, 'mask': mask,
            'sem_vec': txt_out['sem_vec'],   
            'attr_vec': txt_out['attr_vec'], 
            'pred_box': txt_out['pred_box'], 
            'gate': txt_out['gate']          
        }