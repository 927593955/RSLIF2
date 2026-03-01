import torch
import torch.nn as nn
import torch.nn.functional as F

# 保持引用本地模块
from models.yolo_backbone import YOLOv8Backbone, C2f, Conv, Detect
from models.text_encoder import TriStreamDeBERTa

# ==============================================================================
# 1. 核心改进：空间+通道双重门控融合 (Spatial-Channel Gated Fusion)
# ==============================================================================

class ReasoningFusion(nn.Module):
    def __init__(self, visual_dim, text_dim, num_heads=4, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.LayerNorm(visual_dim)
        )
        self.attn = nn.MultiheadAttention(
            visual_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm_v = nn.LayerNorm(visual_dim)
        self.norm_t = nn.LayerNorm(visual_dim)
        
        self.spatial_reasoning = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, kernel_size=7, padding=3, groups=visual_dim),
            nn.BatchNorm2d(visual_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(visual_dim, visual_dim, kernel_size=1),
        )
        self.reasoning_drop = nn.Dropout2d(proj_dropout)
        self.post_norm = nn.BatchNorm2d(visual_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1)) 

    def forward(self, visual_feat, text_feat):
        """
        visual_feat: [B, C, H, W]
        text_feat: 如果是 [B, D] (2D), 会自动升维到 [B, 1, D] 以满足 Attention
        """
        b, c, h, w = visual_feat.shape
        shortcut = visual_feat
        
        # 维度检查与转换: 将 2D [B, D] 转为 3D [B, 1, D]
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
            
        # --- 语义对齐 ---
        v_flat = visual_feat.flatten(2).transpose(1, 2) # [B, HW, C]
        v_flat = self.norm_v(v_flat)
        
        t_feat = self.text_proj(text_feat) # [B, L, C]
        t_feat = self.norm_t(t_feat)
        
        # Cross-Attention
        attn_out, _ = self.attn(query=v_flat, key=t_feat, value=t_feat)
        
        v_aligned = attn_out.transpose(1, 2).view(b, c, h, w)
        v_reasoned = self.spatial_reasoning(v_aligned)
        v_reasoned = self.reasoning_drop(v_reasoned)
        fused = shortcut + self.alpha * v_reasoned
        return self.post_norm(fused)

class CoordGatedFusion(nn.Module):
    """用于 P3 的超轻量级坐标融合"""
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        # 使用线性层而非卷积来生成门控，减少显存
        self.t_norm = nn.LayerNorm(text_dim)
        self.gate_proj = nn.Linear(text_dim, visual_dim)
        self.gate_temp = nn.Parameter(torch.tensor(1.0))
        # 简单的坐标注入
        self.coord_conv = nn.Conv2d(visual_dim + 2, visual_dim, 1)
        self.coord_norm = nn.BatchNorm2d(visual_dim)
        self._coord_cache = {}

    def forward(self, x, t_vec):
        b, c, h, w = x.shape
        # 坐标注入逻辑（缓存网格，避免重复构建）
        key = (h, w, x.device.type, x.device.index, x.dtype)
        if key not in self._coord_cache:
            y_range = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
            x_range = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
            grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
            self._coord_cache[key] = torch.stack([grid_x, grid_y]).unsqueeze(0)
        coords = self._coord_cache[key].expand(b, -1, -1, -1)

        x = self.coord_conv(torch.cat([x, coords], dim=1))
        x = self.coord_norm(x)

        # 通道门控（残差式，避免过抑制）
        t_vec = self.t_norm(t_vec)
        gate_logits = self.gate_proj(t_vec)
        gate = torch.tanh(gate_logits / self.gate_temp.clamp(min=0.1)).view(b, c, 1, 1)
        return x * (1.0 + gate)

class GatedFusion1(nn.Module):
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
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.beta = nn.Parameter(torch.tensor(0.1))

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
        fused = self.fusion_conv(gated_feat)
        return fused + self.beta * visual_feat

# ==============================================================================
# 2. 文本特征池化
# ==============================================================================
class TextFeaturePyramid(nn.Module):
    def __init__(self, text_dim=1024):
        super().__init__()
        # 先做一次共享语义提炼，再分别投影到 P3/P4/P5 对应语义子空间
        self.shared_mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim)
        )
        self.level_proj = nn.ModuleDict({
            'p3': nn.Sequential(nn.Linear(text_dim, text_dim), nn.LayerNorm(text_dim)),
            'p4': nn.Sequential(nn.Linear(text_dim, text_dim), nn.LayerNorm(text_dim)),
            'p5': nn.Sequential(nn.Linear(text_dim, text_dim), nn.LayerNorm(text_dim)),
        })

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

        base = self.shared_mlp(global_text)
        return {
            'global': base,
            'p3': self.level_proj['p3'](base),
            'p4': self.level_proj['p4'](base),
            'p5': self.level_proj['p5'](base)
        }

class SpatialRelationReasoner(nn.Module):
    """
    Geometry-aware spatial-text relation module inspired by spatial VLM designs:
    - inject explicit 2D coordinates
    - perform cross-attention from visual locations to spatial text tokens
    - predict a relation gate to refine spatially-sensitive regions
    """
    def __init__(self, visual_dim, text_dim, num_heads=4):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.coord_proj = nn.Conv2d(2, visual_dim, kernel_size=1, bias=False)
        self.q_norm = nn.LayerNorm(visual_dim)
        self.k_norm = nn.LayerNorm(visual_dim)
        self.attn = nn.MultiheadAttention(visual_dim, num_heads=num_heads, batch_first=True)
        self.refine = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1, groups=visual_dim, bias=False),
            nn.BatchNorm2d(visual_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(visual_dim, visual_dim, kernel_size=1, bias=False)
        )
        self.rel_gate = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.mix = nn.Parameter(torch.tensor(0.25))
        self._coord_cache = {}

    def _coord_grid(self, b, h, w, device, dtype):
        key = (h, w, device.type, device.index, dtype)
        if key not in self._coord_cache:
            y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
            x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            self._coord_cache[key] = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return self._coord_cache[key].expand(b, -1, -1, -1)

    def forward(self, feat, spa_tokens, mask=None):
        b, c, h, w = feat.shape
        coords = self._coord_grid(b, h, w, feat.device, feat.dtype)
        query_feat = feat + self.coord_proj(coords)

        q = query_feat.flatten(2).transpose(1, 2)
        q = self.q_norm(q)

        if spa_tokens.dim() == 2:
            spa_tokens = spa_tokens.unsqueeze(1)
        k = self.k_norm(self.text_proj(spa_tokens))

        key_padding_mask = (mask == 0) if mask is not None else None
        attn_out, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask)
        attn_map = attn_out.transpose(1, 2).reshape(b, c, h, w)

        reasoned = self.refine(attn_map)
        gate = self.rel_gate(reasoned)
        return feat + torch.tanh(self.mix) * gate * reasoned


class GroundingQueryInjector(nn.Module):
    """
    Grounding-DINO style global query injector:
    uses learnable queries + text-guided conditioning to modulate multi-scale features.
    """
    def __init__(self, in_channels, text_dim, num_queries=8, num_heads=4):
        super().__init__()
        self.text_fuse = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim)
        )
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, text_dim) * 0.02)
        self.decoder = nn.MultiheadAttention(text_dim, num_heads=num_heads, batch_first=True)

        self.feat_proj = nn.ModuleList([
            nn.Conv2d(c, text_dim, kernel_size=1) for c in in_channels
        ])
        self.out_heads = nn.ModuleList([
            nn.Linear(text_dim, c * 2) for c in in_channels
        ])

    def forward(self, feats, sem_vec, spa_vec):
        b = sem_vec.shape[0]
        text_seed = self.text_fuse(torch.cat([sem_vec, spa_vec], dim=-1))
        queries = self.query_embed.expand(b, -1, -1) + text_seed.unsqueeze(1)

        vis_tokens = []
        for feat, proj in zip(feats, self.feat_proj):
            token = proj(feat).flatten(2).mean(-1)
            vis_tokens.append(token)
        vis_tokens = torch.stack(vis_tokens, dim=1)

        decoded, _ = self.decoder(queries, vis_tokens, vis_tokens)
        grounding_vec = decoded.mean(dim=1)

        out_feats = []
        for feat, head in zip(feats, self.out_heads):
            gb = head(grounding_vec)
            gamma, beta = torch.chunk(gb, 2, dim=1)
            gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            out_feats.append(feat * (1.0 + gamma) + beta)

        return out_feats, grounding_vec

class ScaleAwareFiLM(nn.Module):
    """Shared MLP + per-level FiLM (scale, shift) heads."""
    def __init__(self, text_dim, level_channels, hidden_ratio=0.5):
        super().__init__()
        hidden_dim = max(int(text_dim * hidden_ratio), 128)
        self.shared = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        self.gamma_heads = nn.ModuleDict({
            level: nn.Linear(hidden_dim, c) for level, c in level_channels.items()
        })
        self.beta_heads = nn.ModuleDict({
            level: nn.Linear(hidden_dim, c) for level, c in level_channels.items()
        })

    def forward(self, feat, text_vec, level):
        h = self.shared(text_vec)
        gamma = self.gamma_heads[level](h).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_heads[level](h).unsqueeze(-1).unsqueeze(-1)
        # 残差式 FiLM：避免纯乘法门控过度抑制
        return feat * (1.0 + torch.tanh(gamma)) + beta

# ==============================================================================
# 3. 深度融合 Neck (PAFPN)
# ==============================================================================
class DeepFusionPAFPN(nn.Module):
    def __init__(self, in_channels=[128, 256, 512], text_dim=1024, use_relation_reasoner=False):
        super().__init__()
        c3, c4, c5 = in_channels
        self.text_pooler = TextFeaturePyramid(text_dim)
        self.attr_film = ScaleAwareFiLM(
            text_dim=text_dim,
            level_channels={'p3': c3, 'p4': c4, 'p5': c5}
        )

        # 每一层级都注入空间/通道门控
        self.fuse_p5 = ReasoningFusion(c5, text_dim) 
        self.fuse_p4 = ReasoningFusion(c4, text_dim, num_heads=4, attn_dropout=0.1, proj_dropout=0.1)
        self.fuse_p3 = CoordGatedFusion(c3, text_dim)
        self.fuse_n4 = CoordGatedFusion(c4, text_dim)
        self.fuse_n5 = ReasoningFusion(c5, text_dim)

        self.use_relation_reasoner = use_relation_reasoner
        if self.use_relation_reasoner:
            self.rel_reason_p3 = SpatialRelationReasoner(c3, text_dim, num_heads=4)
            self.rel_reason_n4 = SpatialRelationReasoner(c4, text_dim, num_heads=4)
            self.rel_reason_n5 = SpatialRelationReasoner(c5, text_dim, num_heads=8)

        self.reduce_p5 = Conv(c5, c4, 1, 1) 
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p4 = C2f(c4, c4, n=3, shortcut=False) 

        self.reduce_p4 = Conv(c4, c3, 1, 1)
        self.c2f_p3 = C2f(c3, c3, n=3, shortcut=False)

        self.down_p3 = Conv(c3, c3, 3, 2)
        self.c2f_n4 = C2f(c3 + c4, c4, n=3, shortcut=False) 
        
        self.down_p4 = Conv(c4, c4, 3, 2)
        self.c2f_n5 = C2f(c4 + c5, c5, n=3, shortcut=False) 

    def forward(self, raw_p3, raw_p4, raw_p5, sem_text, spa_text, attr_text, mask):
        # 1. 获取全局文本向量 [B, 1024]
        sem_pyr = self.text_pooler(sem_text, mask)
        spa_pyr = self.text_pooler(spa_text, mask)
        attr_pyr = self.text_pooler(attr_text, mask)
        sem_vec, spa_vec = sem_pyr['global'], spa_pyr['global']

        # 2. 自顶向下路径 (语义注入)
        f5 = self.fuse_p5(raw_p5, sem_vec) 
        f5_up = self.up(self.reduce_p5(f5))

        f4 = self.c2f_p4(f5_up + self.fuse_p4(raw_p4, spa_pyr['p4']))
        f4_up = self.up(self.reduce_p4(f4))
        
        f3 = self.c2f_p3(f4_up + self.fuse_p3(raw_p3, spa_vec))

        # 3. 自底向上路径 (定位强化)
        n3_down = self.down_p3(f3)
        n4 = self.c2f_n4(torch.cat([n3_down, f4], dim=1))
        n4 = self.fuse_n4(n4, spa_vec) 

        n4_down = self.down_p4(n4)
        n5 = self.c2f_n5(torch.cat([n4_down, f5], dim=1))
        n5 = self.fuse_n5(n5, sem_vec) 

        # 4. 显式空间关系推理（位置敏感）
        if self.use_relation_reasoner:
            f3 = self.rel_reason_p3(f3, spa_text, mask)
            n4 = self.rel_reason_n4(n4, spa_text, mask)
            n5 = self.rel_reason_n5(n5, spa_text, mask)

        # 4. 属性门控 (细粒度属性调制)
        f3 = self.attr_film(f3, attr_pyr['p3'], level='p3')
        n4 = self.attr_film(n4, attr_pyr['p4'], level='p4')
        n5 = self.attr_film(n5, attr_pyr['p5'], level='p5')

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
        self.backbone.load_pretrained_weights(cfg.get('yolo_weight') or 'yolov8s.pt')
        self.use_relation_reasoner = cfg.get('use_relation_reasoner', False)
        self.use_query_injector = cfg.get('use_query_injector', False)
        self.use_cls_text_bias = cfg.get('use_cls_text_bias', False)
        self.neck = DeepFusionPAFPN(
            in_channels=self.backbone.out_channels,
            text_dim=self.text_dim,
            use_relation_reasoner=self.use_relation_reasoner
        )

        nc = cfg.get('nc', 20)
        self.head = Detect(nc=nc, ch=self.backbone.out_channels)
        self.head.stride = torch.tensor([8.0, 16.0, 32.0])
        
        # 视觉-文本对齐投影层 (对齐最终特征图与语义分支向量)
        self.vis_projector = nn.Conv2d(self.backbone.out_channels[-1], self.text_dim, kernel_size=1)

        # Grounding query injector: text-conditioned global querying for multi-scale refinement
        self.query_injector = GroundingQueryInjector(self.backbone.out_channels, self.text_dim, num_queries=8) if self.use_query_injector else None

        # 语义向量对分类 logits 的先验偏置（增强类别敏感性）
        self.cls_text_bias = nn.Linear(self.text_dim, nc) if self.use_cls_text_bias else None
        self.cls_bias_scale = nn.Parameter(torch.tensor(0.1)) if self.use_cls_text_bias else None

        # 初始化
        self._init_custom_modules()
        self.backbone.load_pretrained_weights(cfg.get('yolo_weight') or 'yolov8s.pt')
        self._initialize_biases(self.head)


    def _init_custom_modules(self):
        modules = [self.neck, self.head, self.vis_projector]
        if self.query_injector is not None:
            modules.append(self.query_injector)
        if self.cls_text_bias is not None:
            modules.append(self.cls_text_bias)
        for module in modules:
            module.apply(self._init_weights)

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
        sem_vec = txt_out['sem_vec']
        spa_vec = txt_out['spa_vec']
        relation_vec = txt_out.get('relation_vec', spa_vec)
        attr_vec = txt_out['attr_vec']
        
        # 2. 视觉编码 + 深度融合
        raw_feats = self.backbone(imgs) 
        fused_feats = self.neck(raw_feats[0], raw_feats[1], raw_feats[2], sem, spa, attr, mask)

        # 2.1 Grounding query 注入（替代简单 FiLM，提升关系推理）
        if self.use_query_injector:
            mod_feats, grounding_vec = self.query_injector(fused_feats, sem_vec, relation_vec)
        else:
            mod_feats, grounding_vec = fused_feats, relation_vec

        # 3. 对齐特征提取
        vis_align_feat = self.vis_projector(mod_feats[-1])

        # 4. 检测头输出
        head_output = self.head(mod_feats)

        # 4.1 语义类别偏置注入到分类 logits
        if self.use_cls_text_bias:
            cls_bias = self.cls_text_bias(sem_vec).unsqueeze(-1).unsqueeze(-1)
            bias_scale = torch.tanh(self.cls_bias_scale)
            head_output = [
                torch.cat([lvl[:, :-self.head.nc], lvl[:, -self.head.nc:] + bias_scale * cls_bias], dim=1)
                for lvl in head_output
            ]

        return {
            'feats': head_output, 
            'visual_feats': [vis_align_feat], 
            'sem': sem, 'spa': spa, 'attr': attr, 'mask': mask,
            'sem_vec': sem_vec,   
            'attr_vec': attr_vec, 
            'spa_vec': spa_vec,
            'relation_vec': relation_vec,
            'pred_box': txt_out['pred_box'], 
            'gate': txt_out['gate']          
        }